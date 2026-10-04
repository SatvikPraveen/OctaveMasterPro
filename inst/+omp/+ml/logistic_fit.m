function model = logistic_fit (X, y, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{model} =} omp.ml.logistic_fit (@var{X}, @var{y})
  ## @deftypefnx {} {@var{model} =} omp.ml.logistic_fit (@var{X}, @var{y}, "Lambda", @var{lambda}, "Standardize", @var{tf}, "MaxIter", @var{n}, "Tol", @var{tol})
  ## L2-regularized logistic regression fitted by iteratively reweighted
  ## least squares (Newton--Raphson on the penalized log-likelihood).
  ##
  ## Minimizes
  ## -sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + (lambda/2) ||beta||^2
  ## with p_i = sigma(b0 + x_i' beta).  The intercept b0 is not penalized.
  ## Because the objective is strictly convex for lambda > 0, Newton's
  ## method converges quadratically from beta = 0; a step-halving
  ## safeguard guarantees monotone decrease.  With lambda = 0 and perfectly
  ## separable data the MLE does not exist; a warning is issued if the
  ## iteration fails to converge.
  ##
  ## With @var{Standardize} true (default) feature means and standard
  ## deviations are estimated from @var{X} -- the training data -- and
  ## stored in the model, so @code{omp.ml.logistic_predict} applies the
  ## identical transform to test data.  This prevents the common leakage
  ## of standardizing before splitting.
  ##
  ## Defaults: lambda = 1e-6 (numerical stabilizer), MaxIter = 100,
  ## Tol = 1e-10 on the Newton decrement.
  ## @end deftypefn

  opts = struct ("lambda", 1e-6, "standardize", true, "maxiter", 100, ...
                 "tol", 1e-10);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.ml.logistic_fit: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor
  y = double (y(:));
  [n, d] = size (X);
  if (numel (y) != n)
    error ("omp.ml.logistic_fit: X and y must have the same number of rows");
  endif
  if (any (y != 0 & y != 1))
    error ("omp.ml.logistic_fit: y must be binary (0/1)");
  endif
  if (any (! isfinite (X(:))))
    error ("omp.ml.logistic_fit: X contains NaN or Inf; impute first");
  endif

  mu = zeros (1, d);
  sd = ones (1, d);
  if (opts.standardize)
    mu = mean (X, 1);
    sd = std (X, 0, 1);
    sd(sd == 0) = 1;
  endif
  Z = [ones(n, 1), (X - mu) ./ sd];
  pen = opts.lambda * [0; ones(d, 1)];

  w = zeros (d + 1, 1);
  ## Start the intercept at the log-odds of the base rate.
  pbar = min (max (mean (y), 1e-6), 1 - 1e-6);
  w(1) = log (pbar / (1 - pbar));
  obj = objective (Z, y, w, pen);
  converged = false;
  for it = 1:opts.maxiter
    eta = Z * w;
    p = sigmoid (eta);
    g = Z' * (p - y) + pen .* w;
    W = p .* (1 - p);
    H = Z' * (Z .* W) + diag (pen) + 1e-12 * eye (d + 1);
    step = -(H \ g);
    dec = -(g' * step);                 # Newton decrement squared
    t = 1;
    while (true)
      wn = w + t * step;
      on = objective (Z, y, wn, pen);
      if (on <= obj + 1e-4 * t * (g' * step) || t < 1e-10)
        break;
      endif
      t /= 2;
    endwhile
    w = wn;
    obj = on;
    if (dec / 2 < opts.tol)
      converged = true;
      break;
    endif
  endfor
  if (! converged)
    warning ("omp:logistic_fit:noconv", ...
             "omp.ml.logistic_fit: no convergence in %d iterations (separable data? increase Lambda)", ...
             opts.maxiter);
  endif

  model = struct ("intercept", w(1), "coef", w(2:end), "mu", mu, ...
                  "sigma", sd, "lambda", opts.lambda, "iterations", it, ...
                  "converged", converged, "neg_log_lik", obj);
endfunction

function p = sigmoid (z)
  p = 1 ./ (1 + exp (-z));
endfunction

function v = objective (Z, y, w, pen)
  eta = Z * w;
  ## log(1 + exp(eta)) computed stably.
  sp = max (eta, 0) + log1p (exp (-abs (eta)));
  v = sum (sp - y .* eta) + 0.5 * sum (pen .* w .^ 2);
endfunction

%!test
%! ## Recovers the generating coefficients (large n, weak penalty).
%! omp.repro.seed (1);
%! n = 20000; X = randn (n, 3); b = [1.5; -2; 0]; b0 = -0.5;
%! y = rand (n, 1) < 1 ./ (1 + exp (-(b0 + X * b)));
%! m = omp.ml.logistic_fit (X, y, "Standardize", false);
%! assert (m.converged);
%! assert (m.coef, b, 0.08);
%! assert (m.intercept, b0, 0.05);

%!test
%! ## First-order optimality: penalized score is zero at the solution.
%! omp.repro.seed (2);
%! X = randn (300, 4); y = rand (300, 1) < 0.3;
%! lam = 2;
%! m = omp.ml.logistic_fit (X, y, "Lambda", lam);
%! Z = [ones(300, 1), (X - m.mu) ./ m.sigma];
%! w = [m.intercept; m.coef];
%! p = 1 ./ (1 + exp (-Z * w));
%! g = Z' * (p - y) + lam * [0; w(2:end)];
%! assert (norm (g, Inf) < 1e-6);

%!test
%! ## Separable data stay finite with a ridge penalty.
%! X = [-2; -1; 1; 2]; y = [0; 0; 1; 1];
%! m = omp.ml.logistic_fit (X, y, "Lambda", 0.1);
%! assert (m.converged);
%! assert (all (isfinite (m.coef)));

%!test
%! ## Standardization statistics come from the training data only.
%! X = [1 10; 2 20; 3 30; 4 40]; y = [0; 1; 0; 1];
%! m = omp.ml.logistic_fit (X, y);
%! assert (m.mu, [2.5 25]);
%! assert (m.sigma, std (X));

%!error <binary> omp.ml.logistic_fit ([1; 2], [0; 2])
%!error <NaN or Inf> omp.ml.logistic_fit ([1; NaN], [0; 1])
