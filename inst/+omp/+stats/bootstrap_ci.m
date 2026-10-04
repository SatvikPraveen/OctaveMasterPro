function [ci, theta, boot] = bootstrap_ci (data, statfn, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{ci}, @var{theta}, @var{boot}] =} omp.stats.bootstrap_ci (@var{data}, @var{statfn})
  ## @deftypefnx {} {[@dots{}] =} omp.stats.bootstrap_ci (@dots{}, "B", @var{B}, "Alpha", @var{alpha}, "Method", @var{m}, "Seed", @var{s}, "Vectorized", @var{tf})
  ## Nonparametric bootstrap confidence interval for the scalar statistic
  ## @code{statfn (data)}.
  ##
  ## Rows of @var{data} are the resampling units (a column vector is n
  ## observations).  Resampling rows keeps multivariate observations, e.g.
  ## (prediction, label) pairs, intact.
  ##
  ## @var{m} is @qcode{"bca"} (default) or @qcode{"percentile"}.  BCa
  ## (Efron, 1987; Efron & Tibshirani, 1993, Ch. 14) corrects for median
  ## bias z0 and skewness via the jackknife acceleration a; it is
  ## second-order accurate, coverage error O(1/n) versus O(1/sqrt(n)) for
  ## the percentile interval.
  ##
  ## Defaults: B = 2000, alpha = 0.05 (two-sided 95% interval).  With
  ## @var{Vectorized} true and vector @var{data}, @var{statfn} receives an
  ## n-by-B matrix of resamples and must return a 1-by-B row (much faster).
  ##
  ## Returns @var{ci} = [lower, upper], the point estimate @var{theta} and
  ## the bootstrap replicates @var{boot}.
  ##
  ## Caveat: rows must be exchangeable.  For time series or clustered data
  ## resample blocks or clusters instead; the i.i.d. bootstrap otherwise
  ## underestimates variance.
  ## @end deftypefn

  opts = struct ("b", 2000, "alpha", 0.05, "method", "bca", "seed", [], ...
                 "vectorized", false);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.stats.bootstrap_ci: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor
  if (isrow (data))
    data = data(:);
  endif
  n = rows (data);
  if (n < 2)
    error ("omp.stats.bootstrap_ci: need at least 2 observations");
  endif
  if (! isempty (opts.seed))
    state = omp.repro.seed (opts.seed);
    restore = onCleanup (@() omp.repro.seed (state));
  endif

  theta = statfn (data);
  idx = randi (n, n, opts.b);
  if (opts.vectorized)
    if (columns (data) != 1)
      error ("omp.stats.bootstrap_ci: Vectorized requires vector DATA");
    endif
    boot = statfn (data(idx))(:);
  else
    boot = zeros (opts.b, 1);
    for j = 1:opts.b
      boot(j) = statfn (data(idx(:, j), :));
    endfor
  endif
  boot = boot(isfinite (boot));

  lo = opts.alpha / 2;
  hi = 1 - opts.alpha / 2;
  switch (lower (opts.method))
    case "percentile"
      ci = quantile7 (boot, [lo hi]);

    case "bca"
      ## Bias correction; ties counted half so a degenerate (constant)
      ## bootstrap distribution gives z0 = 0 rather than +/-Inf.
      z0 = norminv_ (mean (boot < theta) + 0.5 * mean (boot == theta));
      ## Jackknife acceleration.
      jack = zeros (n, 1);
      for i = 1:n
        jack(i) = statfn (data([1:i-1, i+1:n], :));
      endfor
      d = mean (jack) - jack;
      den = 6 * sum (d .^ 2) ^ 1.5;
      if (den > 0)
        a = sum (d .^ 3) / den;
      else
        a = 0;
      endif
      z = norminv_ ([lo hi]);
      adj = normcdf_ (z0 + (z0 + z) ./ (1 - a * (z0 + z)));
      if (any (! isfinite (adj)))
        adj = [lo hi];
      endif
      ci = quantile7 (boot, adj);

    otherwise
      error ("omp.stats.bootstrap_ci: unknown method '%s'", opts.method);
  endswitch
endfunction

function q = quantile7 (x, p)
  x = sort (x(:));
  n = numel (x);
  h = (n - 1) * p(:)' + 1;
  lo = floor (h);
  hi = min (lo + 1, n);
  q = x(lo)' + (h - lo) .* (x(hi)' - x(lo)');
endfunction

function z = norminv_ (p)
  z = -sqrt (2) * erfcinv (2 * p);
endfunction

function p = normcdf_ (z)
  p = 0.5 * erfc (-z / sqrt (2));
endfunction

%!test
%! ## Point estimate and ordering.
%! x = (1:20)';
%! [ci, th] = omp.stats.bootstrap_ci (x, @mean, "Seed", 1);
%! assert (th, 10.5);
%! assert (ci(1) < th && th < ci(2));

%!test
%! ## Reproducible with a seed, and percentile ~ BCa for symmetric data.
%! omp.repro.seed (3); x = randn (200, 1);
%! a = omp.stats.bootstrap_ci (x, @mean, "Seed", 5, "Method", "percentile");
%! b = omp.stats.bootstrap_ci (x, @mean, "Seed", 5, "Method", "percentile");
%! c = omp.stats.bootstrap_ci (x, @mean, "Seed", 5, "Method", "bca");
%! assert (a, b);
%! assert (c, a, 0.02);
%! ## Width close to the normal-theory interval 2*1.96*s/sqrt(n).
%! assert (diff (a), 2 * 1.96 * std (x) / sqrt (200), 0.03);

%!test
%! ## Vectorized and loop paths agree exactly for the same seed.
%! x = (1:15)' .^ 2;
%! [a, ~, ba] = omp.stats.bootstrap_ci (x, @mean, "Seed", 2, "Method", "percentile");
%! [b, ~, bb] = omp.stats.bootstrap_ci (x, @(m) mean (m, 1), "Seed", 2, ...
%!                                      "Method", "percentile", "Vectorized", true);
%! assert (ba, bb, 1e-12);
%! assert (a, b, 1e-12);

%!test
%! ## Monte Carlo coverage of the BCa interval for an exponential mean
%! ## (skewed), n = 40.  Nominal 0.90; accept a +/-3.5 SE band around 0.88
%! ## (small-sample BCa slightly undercovers).
%! omp.repro.seed (10);
%! R = 300; hit = 0;
%! for r = 1:R
%!   x = rande (40, 1);
%!   ci = omp.stats.bootstrap_ci (x, @(m) mean (m, 1), "B", 999, ...
%!                                "Alpha", 0.10, "Vectorized", true);
%!   hit += (ci(1) <= 1 && 1 <= ci(2));
%! endfor
%! cov = hit / R;
%! assert (abs (cov - 0.88) < 3.5 * sqrt (0.9 * 0.1 / R));

%!test
%! ## Constant data: degenerate but finite interval.
%! ci = omp.stats.bootstrap_ci (ones (10, 1), @mean, "B", 200);
%! assert (ci, [1 1]);

%!test
%! ## Matrix data: rows resampled jointly.  For a bivariate-normal
%! ## correlation the BCa interval should agree with Fisher's z interval.
%! ## (Checking that one interval covers the true rho would fail 5% of the
%! ## time by construction; coverage is tested by simulation above.)
%! omp.repro.seed (4);
%! z = randn (300, 1); D = [z, z + 0.5 * randn(300, 1)];
%! r = @(M) corr (M(:, 1), M(:, 2));
%! [ci, th] = omp.stats.bootstrap_ci (D, r, "B", 2000, "Seed", 1);
%! fisher = tanh (atanh (th) + [-1 1] * 1.959964 / sqrt (300 - 3));
%! assert (ci, fisher, 0.01);

%!error <unknown method> omp.stats.bootstrap_ci ((1:5)', @mean, "Method", "foo")
%!error <at least 2> omp.stats.bootstrap_ci (1, @mean)
