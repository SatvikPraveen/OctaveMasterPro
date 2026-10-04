function [alpha, f, g, nfev, ok] = line_search_wolfe (fg, x, p, f0, g0, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{alpha}, @var{f}, @var{g}, @var{nfev}, @var{ok}] =} omp.optim.line_search_wolfe (@var{fg}, @var{x}, @var{p}, @var{f0}, @var{g0})
  ## @deftypefnx {} {[@dots{}] =} omp.optim.line_search_wolfe (@dots{}, "c1", @var{c1}, "c2", @var{c2}, "AlphaInit", @var{a0}, "MaxEval", @var{n})
  ## Find a step length satisfying the strong Wolfe conditions
  ## @example
  ## f(x + a p) <= f(x) + c1 a g'p              (sufficient decrease)
  ## |g(x + a p)' p| <= c2 |g'p|                (curvature)
  ## @end example
  ## by the bracketing/zoom procedure of Nocedal & Wright,
  ## @cite{Numerical Optimization}, 2nd ed., Algorithms 3.5 and 3.6, with
  ## safeguarded cubic interpolation inside the zoom phase.
  ##
  ## @var{fg} returns @code{[f, g]} at a point.  @var{p} must be a descent
  ## direction (g0'p < 0).  Defaults: c1 = 1e-4, c2 = 0.9 (quasi-Newton
  ## setting), initial step 1, at most 50 function evaluations.
  ##
  ## @var{ok} is false if no acceptable step was found; the best
  ## sufficient-decrease point seen is then returned.
  ## @end deftypefn

  opts = struct ("c1", 1e-4, "c2", 0.9, "alphainit", 1, "maxeval", 50, ...
                 "alphamax", 1e10);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.optim.line_search_wolfe: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor

  d0 = g0' * p;
  if (d0 >= 0)
    error ("omp.optim.line_search_wolfe: P is not a descent direction");
  endif
  c1 = opts.c1;
  c2 = opts.c2;

  nfev = 0;
  a_prev = 0; f_prev = f0; d_prev = d0;
  a = opts.alphainit;
  ok = false;
  while (nfev < opts.maxeval)
    [f, g] = fg (x + a * p);
    nfev += 1;
    d = g' * p;
    if (f > f0 + c1 * a * d0 || (nfev > 1 && f >= f_prev))
      [alpha, f, g, nfev, ok] = zoom (a_prev, f_prev, d_prev, a, f, d, nfev);
      return;
    endif
    if (abs (d) <= -c2 * d0)
      alpha = a; ok = true;
      return;
    endif
    if (d >= 0)
      [alpha, f, g, nfev, ok] = zoom (a, f, d, a_prev, f_prev, d_prev, nfev);
      return;
    endif
    a_prev = a; f_prev = f; d_prev = d;
    a = min (2 * a, opts.alphamax);
  endwhile
  alpha = a;

  function [alpha, f, g, nfev, ok] = zoom (alo, flo, dlo, ahi, fhi, dhi, nfev)
    ## Invariant: alo satisfies sufficient decrease and has the lowest f
    ## among trial points; (ahi - alo) * dlo < 0.
    ok = false;
    alpha = alo; f = flo; g = [];
    while (nfev < opts.maxeval)
      aj = cubic_min (alo, flo, dlo, ahi, fhi, dhi);
      lo = min (alo, ahi); hi = max (alo, ahi); w = hi - lo;
      if (! isfinite (aj) || aj < lo + 0.1 * w || aj > hi - 0.1 * w)
        aj = (alo + ahi) / 2;       # safeguard: bisection
      endif
      [fj, gj] = fg (x + aj * p);
      nfev += 1;
      dj = gj' * p;
      if (fj > f0 + c1 * aj * d0 || fj >= flo)
        ahi = aj; fhi = fj; dhi = dj;
      else
        if (abs (dj) <= -c2 * d0)
          alpha = aj; f = fj; g = gj; ok = true;
          return;
        endif
        if (dj * (ahi - alo) >= 0)
          ahi = alo; fhi = flo; dhi = dlo;
        endif
        alo = aj; flo = fj; dlo = dj;
      endif
      if (abs (ahi - alo) < eps * max (1, abs (alo)))
        break;
      endif
    endwhile
    alpha = alo; f = flo;
    [~, g] = fg (x + alo * p);
    nfev += 1;
  endfunction
endfunction

function a = cubic_min (a0, f0, d0, a1, f1, d1)
  ## Minimizer of the cubic interpolating (f, f') at a0 and a1
  ## (Nocedal & Wright, eq. 3.59).
  d1_ = d0 + d1 - 3 * (f0 - f1) / (a0 - a1);
  disc = d1_ ^ 2 - d0 * d1;
  if (disc < 0)
    a = NaN;
    return;
  endif
  d2 = sign (a1 - a0) * sqrt (disc);
  a = a1 - (a1 - a0) * (d1 + d2 - d1_) / (d1 - d0 + 2 * d2);
endfunction

%!function [f, g] = quad1 (x)
%!  f = (x - 3) ^ 2;
%!  g = 2 * (x - 3);
%!endfunction

%!function [f, g] = rosen (x)
%!  f = 100 * (x(2) - x(1) ^ 2) ^ 2 + (1 - x(1)) ^ 2;
%!  g = [-400 * x(1) * (x(2) - x(1) ^ 2) - 2 * (1 - x(1)); 200 * (x(2) - x(1) ^ 2)];
%!endfunction

%!test
%! [f0, g0] = quad1 (0);
%! [a, f, g, n, ok] = omp.optim.line_search_wolfe (@quad1, 0, 1, f0, g0);
%! assert (ok);
%! assert (f <= f0 + 1e-4 * a * g0);
%! assert (abs (g) <= 0.9 * abs (g0));

%!test
%! ## Strong Wolfe conditions hold on Rosenbrock along -grad from many points.
%! omp.repro.seed (4);
%! for t = 1:25
%!   x = 2 * randn (2, 1);
%!   [f0, g0] = rosen (x);
%!   p = -g0;
%!   [a, f, g, n, ok] = omp.optim.line_search_wolfe (@rosen, x, p, f0, g0, "c2", 0.5);
%!   assert (ok);
%!   assert (f <= f0 + 1e-4 * a * g0' * p);
%!   assert (abs (g' * p) <= 0.5 * abs (g0' * p) * (1 + 1e-12));
%! endfor

%!error <not a descent direction> omp.optim.line_search_wolfe (@(x) deal (x^2, 2*x), 1, 1, 1, 2)
