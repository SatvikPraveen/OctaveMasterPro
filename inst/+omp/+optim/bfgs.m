function [x, fval, info] = bfgs (fg, x0, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{x}, @var{fval}, @var{info}] =} omp.optim.bfgs (@var{fg}, @var{x0})
  ## @deftypefnx {} {[@dots{}] =} omp.optim.bfgs (@var{fg}, @var{x0}, "GradTol", @var{gtol}, "MaxIter", @var{n}, "Display", @var{tf})
  ## Minimize a smooth function with the BFGS quasi-Newton method and a
  ## strong-Wolfe line search (Nocedal & Wright, Alg. 6.1).
  ##
  ## @var{fg} returns @code{[f, g]}: the objective value and its gradient
  ## (column vector).  The inverse-Hessian approximation H is initialized
  ## after the first step with the Shanno--Phua scaling
  ## H0 = (s'y / y'y) I (N&W eq. 6.20).  The update is skipped when the
  ## curvature condition s'y > 0 fails numerically, which keeps H positive
  ## definite.
  ##
  ## Termination: ||g||_inf <= GradTol * max(1, |f|) (default 1e-8),
  ## MaxIter iterations (default 1000), or a failed line search.
  ##
  ## @var{info} fields: @code{converged}, @code{iterations}, @code{nfev},
  ## @code{grad_norm}, @code{message}, and the iterate history
  ## @code{trace.f}, @code{trace.gnorm}, @code{trace.x}.
  ## @end deftypefn

  opts = struct ("gradtol", 1e-8, "maxiter", 1000, "display", false);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.optim.bfgs: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor

  x = x0(:);
  n = numel (x);
  [f, g] = fg (x);
  nfev = 1;
  H = eye (n);
  first = true;

  tr.f = f; tr.gnorm = norm (g, Inf); tr.x = x';
  converged = false;
  message = "maximum iterations reached";
  it = 0;
  while (it < opts.maxiter)
    if (norm (g, Inf) <= opts.gradtol * max (1, abs (f)))
      converged = true;
      message = "gradient tolerance satisfied";
      break;
    endif
    it += 1;

    p = -H * g;
    if (g' * p >= 0)               # safeguard: reset to steepest descent
      H = eye (n);
      p = -g;
    endif
    a0 = 1;
    if (first)
      a0 = min (1, 1 / norm (g, Inf));
    endif
    [alpha, fn, gn, ne, ok] = omp.optim.line_search_wolfe (fg, x, p, f, g, ...
                                                          "AlphaInit", a0);
    nfev += ne;
    if (! ok && fn >= f)
      message = "line search failed";
      break;
    endif

    s = alpha * p;
    y = gn - g;
    sy = s' * y;
    if (sy > sqrt (eps) * norm (s) * norm (y))
      if (first)
        H = (sy / (y' * y)) * eye (n);
        first = false;
      endif
      rho = 1 / sy;
      Hy = H * y;
      ## H+ = (I - rho s y') H (I - rho y s') + rho s s', expanded.
      H += rho * ((1 + rho * (y' * Hy)) * (s * s') - (Hy * s') - (s * Hy'));
    endif

    x += s; f = fn; g = gn;
    tr.f(end+1, 1) = f;
    tr.gnorm(end+1, 1) = norm (g, Inf);
    tr.x(end+1, :) = x';
    if (opts.display)
      printf ("%5d  f = %-14.6e  |g| = %-10.3e  alpha = %.3g\n", it, f, ...
              norm (g, Inf), alpha);
    endif
  endwhile

  fval = f;
  x = reshape (x, size (x0));
  info = struct ("converged", converged, "iterations", it, "nfev", nfev, ...
                 "grad_norm", norm (g, Inf), "message", message, "trace", tr);
endfunction

%!test
%! ## Rosenbrock from the classical starting point.
%! [x, f, info] = omp.optim.bfgs (@omp.optim.rosenbrock, [-1.2; 1]);
%! assert (info.converged);
%! assert (x, [1; 1], 1e-6);
%! assert (f < 1e-12);
%! assert (info.iterations < 60);
%! assert (all (diff (info.trace.f) <= 0));   # monotone decrease

%!test
%! ## Higher-dimensional Rosenbrock.
%! [x, f, info] = omp.optim.bfgs (@omp.optim.rosenbrock, zeros (10, 1));
%! assert (info.converged);
%! assert (x, ones (10, 1), 1e-5);

%!test
%! ## Strictly convex quadratic: converges to A\b.
%! omp.repro.seed (6);
%! M = randn (8); A = M' * M + eye (8); b = randn (8, 1);
%! fg = @(x) deal (0.5 * x' * A * x - b' * x, A * x - b);
%! [x, ~, info] = omp.optim.bfgs (fg, zeros (8, 1), "GradTol", 1e-12);
%! assert (x, A \ b, 1e-8);

%!test
%! ## Superlinear convergence: error ratios e_{k+1}/e_k -> 0.
%! [~, ~, info] = omp.optim.bfgs (@omp.optim.rosenbrock, [-1.2; 1]);
%! e = sqrt (sum ((info.trace.x - 1) .^ 2, 2));
%! e = e(e > 1e-12);
%! r = e(2:end) ./ e(1:end-1);
%! assert (min (r(end-3:end)) < 0.1);

%!test
%! ## Row-vector starting point keeps its shape.
%! x = omp.optim.bfgs (@omp.optim.rosenbrock, [-1.2 1]);
%! assert (size (x), [1 2]);
