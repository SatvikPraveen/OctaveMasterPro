function [q, C, qk] = convergence_order (e)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{q}, @var{C}, @var{qk}] =} omp.optim.convergence_order (@var{e})
  ## Estimate the order of convergence q and asymptotic constant C of an
  ## error sequence e_k -> 0 obeying e_(k+1) ~ C e_k^q.
  ##
  ## Per-step estimates use three consecutive errors,
  ## q_k = log(e_(k+1)/e_k) / log(e_k/e_(k-1)),
  ## which does not require knowing C.  @var{q} is the median of the last
  ## three finite q_k (robust to the pre-asymptotic phase and to the final
  ## iterate hitting machine precision); @var{C} = e_end / e_(end-1)^q.
  ##
  ## Errors at or below 10 u are discarded: once rounding dominates the
  ## sequence carries no information about the order.
  ## @end deftypefn

  e = abs (e(:));
  e = e(e > 10 * eps);
  if (numel (e) < 3)
    error ("omp.optim.convergence_order: need at least 3 errors above 10*eps");
  endif
  qk = log (e(3:end) ./ e(2:end-1)) ./ log (e(2:end-1) ./ e(1:end-2));
  f = qk(isfinite (qk));
  q = median (f(max (1, end-2):end));
  C = e(end) / e(end-1) ^ q;
endfunction

%!test
%! ## Newton's method for sqrt(2): quadratic, C = 1/(2 sqrt 2).
%! x = 1; e = [];
%! for k = 1:6
%!   x = x - (x ^ 2 - 2) / (2 * x);
%!   e(end+1) = x - sqrt (2);
%! endfor
%! [q, C] = omp.optim.convergence_order (e);
%! assert (q, 2, 0.05);
%! assert (C, 1 / (2 * sqrt (2)), 0.05);

%!test
%! ## Bisection-like linear convergence with C = 1/2.
%! [q, C] = omp.optim.convergence_order (0.5 .^ (1:30));
%! assert (q, 1, 1e-10);
%! assert (C, 0.5, 1e-10);

%!test
%! ## Secant method: golden-ratio order.
%! f = @(x) x .^ 3 - 2;
%! x = [1 2];
%! for k = 1:8
%!   x(end+1) = x(end) - f (x(end)) * (x(end) - x(end-1)) / (f (x(end)) - f (x(end-1)));
%! endfor
%! q = omp.optim.convergence_order (x - 2 ^ (1/3));
%! assert (q, (1 + sqrt (5)) / 2, 0.1);

%!error <at least 3> omp.optim.convergence_order ([1e-20 1e-30])
