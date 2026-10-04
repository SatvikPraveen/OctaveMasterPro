function [U, S, V, info] = randomized_svd (A, k, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{U}, @var{S}, @var{V}] =} omp.linalg.randomized_svd (@var{A}, @var{k})
  ## @deftypefnx {} {[@dots{}] =} omp.linalg.randomized_svd (@var{A}, @var{k}, "Oversample", @var{p}, "PowerIters", @var{q}, "Seed", @var{s})
  ## @deftypefnx {} {[@var{U}, @var{S}, @var{V}, @var{info}] =} omp.linalg.randomized_svd (@dots{})
  ## Rank-@var{k} truncated SVD by randomized range finding
  ## (Halko, Martinsson & Tropp, @cite{SIAM Review} 53(2), 2011,
  ## Algorithms 4.3/4.4 + 5.1).
  ##
  ## A Gaussian test matrix Omega (n-by-(k+p)) sketches the range of
  ## (A A')^q A; the sketch is re-orthonormalized after every application
  ## of A or A' (subspace iteration) to prevent the singular values below
  ## u^(1/(2q+1)) sigma_1 from being lost to rounding.  The SVD of the small
  ## matrix Q' A is then lifted back.
  ##
  ## Expected error (HMT Thm. 10.6, q = 0):
  ## E ||A - U S V'||_F <= (1 + k/(p-1))^(1/2) (sum_(j>k) sigma_j^2)^(1/2).
  ## Power iterations sharpen the bound by driving the effective spectral
  ## decay to sigma_j^(2q+1).
  ##
  ## Defaults: @var{p} = 10, @var{q} = 1.  @var{info} reports the
  ## oversampling, power iterations and sketch size used.
  ## @end deftypefn

  opts = struct ("oversample", 10, "poweriters", 1, "seed", []);
  for j = 1:2:numel (varargin)
    key = lower (varargin{j});
    if (! isfield (opts, key))
      error ("omp.linalg.randomized_svd: unknown option '%s'", varargin{j});
    endif
    opts.(key) = varargin{j+1};
  endfor

  [m, n] = size (A);
  if (! (isscalar (k) && k >= 1 && k == fix (k) && k <= min (m, n)))
    error ("omp.linalg.randomized_svd: K must be an integer in [1, min(size(A))]");
  endif
  l = min (k + opts.oversample, min (m, n));

  if (! isempty (opts.seed))
    state = omp.repro.seed (opts.seed);
    restore = onCleanup (@() omp.repro.seed (state));
  endif

  Omega = randn (n, l);
  [Q, ~] = qr (A * Omega, 0);
  for it = 1:opts.poweriters
    [Z, ~] = qr (A' * Q, 0);
    [Q, ~] = qr (A * Z, 0);
  endfor

  B = Q' * A;                       # l-by-n
  [Ub, S, V] = svd (B, "econ");
  U = Q * Ub(:, 1:k);
  S = S(1:k, 1:k);
  V = V(:, 1:k);

  info = struct ("oversample", l - k, "power_iters", opts.poweriters, ...
                 "sketch_size", l);
endfunction

%!function s = svds_ref (A, k)
%!  s = svd (A);
%!  s = s(1:k);
%!endfunction

%!function e = norm_fro_err (A, k, q)
%!  [U, S, V] = omp.linalg.randomized_svd (A, k, "PowerIters", q, "Seed", 1);
%!  e = norm (A - U * S * V', "fro");
%!endfunction

%!test
%! ## Exact recovery of an exactly rank-k matrix.
%! omp.repro.seed (21);
%! A = randn (200, 6) * randn (6, 80);
%! [U, S, V] = omp.linalg.randomized_svd (A, 6, "PowerIters", 0);
%! assert (norm (A - U * S * V', "fro") / norm (A, "fro") < 1e-12);
%! assert (U' * U, eye (6), 1e-12);
%! assert (V' * V, eye (6), 1e-12);
%! assert (diag (S), svds_ref (A, 6), 1e-10 * norm (A));

%!test
%! ## Near-optimality on a matrix with slow spectral decay, and the
%! ## benefit of power iterations (HMT Sec. 4.5).
%! omp.repro.seed (22);
%! m = 300; n = 200; k = 10;
%! [Uo, ~] = qr (randn (m, n), 0);
%! [Vo, ~] = qr (randn (n));
%! sig = 1 ./ (1:n)' .^ 0.5;
%! A = Uo * diag (sig) * Vo';
%! opt = sqrt (sum (sig(k+1:end) .^ 2));          # Eckart-Young optimum
%! err = @(q) norm_fro_err (A, k, q);
%! e0 = err (0); e2 = err (2);
%! assert (e0 >= opt * (1 - 1e-12));
%! assert (e0 <= sqrt (1 + k / 9) * opt * 1.5);   # HMT bound with slack
%! assert (e2 < e0);
%! assert (e2 < 1.05 * opt);

%!test
%! ## Seed option is reproducible and restores the caller's RNG state.
%! A = magic (20);
%! rand ("state", 3); r0 = rand ();
%! rand ("state", 3);
%! [U1, S1] = omp.linalg.randomized_svd (A, 3, "Seed", 9);
%! r1 = rand ();
%! [U2, S2] = omp.linalg.randomized_svd (A, 3, "Seed", 9);
%! assert (S1, S2);
%! assert (r0, r1);

%!error <K must be> omp.linalg.randomized_svd (eye (3), 4)
