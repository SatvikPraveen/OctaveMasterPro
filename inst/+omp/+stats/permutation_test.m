function [p, t_obs, t_null] = permutation_test (x, y, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{p}, @var{t_obs}, @var{t_null}] =} omp.stats.permutation_test (@var{x}, @var{y})
  ## @deftypefnx {} {[@dots{}] =} omp.stats.permutation_test (@dots{}, "Statistic", @var{fn}, "B", @var{B}, "Tail", @var{tail}, "Paired", @var{tf}, "Seed", @var{s})
  ## Monte Carlo permutation test of the null hypothesis that @var{x} and
  ## @var{y} are exchangeable.
  ##
  ## Unpaired (default): group labels are permuted; the default statistic
  ## is mean(x) - mean(y).  Paired: x and y must have equal length, signs
  ## of the differences d = x - y are flipped at random (null: d is
  ## symmetric about 0); the default statistic is mean(d) and a custom
  ## @var{fn} receives d alone.
  ##
  ## @var{tail}: @qcode{"both"} (default, compares |T|), @qcode{"right"}
  ## or @qcode{"left"}.
  ##
  ## The p-value is (1 + #@{T* at least as extreme as T@}) / (B + 1)
  ## (Phipson & Smyth, 2010), which is exact-valid under the null and never
  ## zero.  Default B = 9999.
  ## @end deftypefn

  opts = struct ("statistic", [], "b", 9999, "tail", "both", ...
                 "paired", false, "seed", []);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.stats.permutation_test: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor
  x = x(:);
  y = y(:);
  if (! isempty (opts.seed))
    state = omp.repro.seed (opts.seed);
    restore = onCleanup (@() omp.repro.seed (state));
  endif

  B = opts.b;
  if (opts.paired)
    if (numel (x) != numel (y))
      error ("omp.stats.permutation_test: paired samples must have equal length");
    endif
    d = x - y;
    S = 2 * (rand (numel (d), B) < 0.5) - 1;
    if (isempty (opts.statistic))
      t_obs = mean (d);
      t_null = mean (S .* d, 1)';
    else
      fn = opts.statistic;
      t_obs = fn (d);
      t_null = zeros (B, 1);
      for b = 1:B
        t_null(b) = fn (S(:, b) .* d);
      endfor
    endif
  else
    z = [x; y];
    nx = numel (x);
    N = numel (z);
    if (isempty (opts.statistic))
      t_obs = mean (x) - mean (y);
      [~, P] = sort (rand (N, B), 1);
      Z = z(P);
      t_null = (mean (Z(1:nx, :), 1) - mean (Z(nx+1:end, :), 1))';
    else
      fn = opts.statistic;
      t_obs = fn (x, y);
      t_null = zeros (B, 1);
      for b = 1:B
        P = randperm (N);
        t_null(b) = fn (z(P(1:nx)), z(P(nx+1:end)));
      endfor
    endif
  endif

  ## Tolerance guards against spurious strictness from rounding when T*
  ## equals T in exact arithmetic (e.g. the identity permutation).
  tol = 1e-12 * max (1, abs (t_obs));
  switch (lower (opts.tail))
    case "both"
      extreme = abs (t_null) >= abs (t_obs) - tol;
    case "right"
      extreme = t_null >= t_obs - tol;
    case "left"
      extreme = t_null <= t_obs + tol;
    otherwise
      error ("omp.stats.permutation_test: TAIL must be both, right or left");
  endswitch
  p = (1 + sum (extreme)) / (B + 1);
endfunction

%!test
%! ## Clear shift is detected; p is bounded below by 1/(B+1).
%! omp.repro.seed (1);
%! x = randn (30, 1) + 2; y = randn (30, 1);
%! p = omp.stats.permutation_test (x, y, "B", 999);
%! assert (p, 1 / 1000);

%!test
%! ## Validity: under H0 the rejection rate at 0.05 is ~0.05.
%! omp.repro.seed (2);
%! R = 300; rej = 0;
%! for r = 1:R
%!   rej += omp.stats.permutation_test (randn (15, 1), randn (15, 1), "B", 199) <= 0.05;
%! endfor
%! assert (abs (rej / R - 0.05) < 3.5 * sqrt (0.05 * 0.95 / R));

%!test
%! ## Paired test detects a consistent within-pair shift that the
%! ## unpaired test misses (large between-subject variance).
%! omp.repro.seed (3);
%! s = 10 * randn (25, 1);
%! x = s + 0.5 + 0.3 * randn (25, 1); y = s + 0.3 * randn (25, 1);
%! pp = omp.stats.permutation_test (x, y, "Paired", true, "B", 1999);
%! pu = omp.stats.permutation_test (x, y, "B", 1999);
%! assert (pp < 0.01);
%! assert (pu > 0.2);

%!test
%! ## Custom statistic and one-sided tails.
%! omp.repro.seed (4);
%! x = randn (20, 1) + 1; y = randn (20, 1);
%! med = @(a, b) median (a) - median (b);
%! pr = omp.stats.permutation_test (x, y, "Statistic", med, "Tail", "right", "B", 999);
%! pl = omp.stats.permutation_test (x, y, "Statistic", med, "Tail", "left", "B", 999);
%! assert (pr < 0.05);
%! assert (pl > 0.9);

%!error <equal length> omp.stats.permutation_test (1:3, 1:4, "Paired", true)
