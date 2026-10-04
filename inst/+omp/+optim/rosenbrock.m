function [f, g, H] = rosenbrock (x)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{f}, @var{g}, @var{H}] =} omp.optim.rosenbrock (@var{x})
  ## Extended Rosenbrock function
  ## f(x) = sum_(i=1)^(n-1) 100 (x_(i+1) - x_i^2)^2 + (1 - x_i)^2
  ## with analytic gradient @var{g} and Hessian @var{H}.
  ## The global minimum is f = 0 at x = (1, @dots{}, 1).
  ## @end deftypefn

  x = x(:);
  n = numel (x);
  if (n < 2)
    error ("omp.optim.rosenbrock: X must have at least 2 elements");
  endif
  a = x(1:n-1);
  b = x(2:n);
  r = b - a .^ 2;
  f = sum (100 * r .^ 2 + (1 - a) .^ 2);
  if (nargout > 1)
    g = zeros (n, 1);
    g(1:n-1) = -400 * a .* r - 2 * (1 - a);
    g(2:n) += 200 * r;
  endif
  if (nargout > 2)
    d0 = zeros (n, 1);
    d0(1:n-1) = 1200 * a .^ 2 - 400 * b + 2;
    d0(2:n) += 200;
    d1 = -400 * a;
    H = diag (d0) + diag (d1, 1) + diag (d1, -1);
  endif
endfunction

%!assert (omp.optim.rosenbrock ([1; 1; 1]), 0)
%!assert (omp.optim.rosenbrock ([-1.2; 1]), 24.2, 1e-12)

%!test
%! omp.repro.seed (8);
%! x = randn (5, 1);
%! [~, g, H] = omp.optim.rosenbrock (x);
%! assert (omp.optim.gradcheck (@omp.optim.rosenbrock, x) < 1e-7);
%! ## Hessian columns against central differences of the gradient.
%! h = 1e-6;
%! for j = 1:5
%!   e = zeros (5, 1); e(j) = h;
%!   [~, gp] = omp.optim.rosenbrock (x + e);
%!   [~, gm] = omp.optim.rosenbrock (x - e);
%!   assert (H(:, j), (gp - gm) / (2 * h), 1e-4 * max (1, norm (H(:, j))));
%! endfor
