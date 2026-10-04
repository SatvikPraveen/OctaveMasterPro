function [relerr, g_fd, g] = gradcheck (fg, x, h)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{relerr} =} omp.optim.gradcheck (@var{fg}, @var{x})
  ## @deftypefnx {} {[@var{relerr}, @var{g_fd}, @var{g}] =} omp.optim.gradcheck (@var{fg}, @var{x}, @var{h})
  ## Compare the analytic gradient returned by @code{[f, g] = fg (x)} with
  ## central finite differences
  ## g_fd(j) = (f(x + h e_j) - f(x - h e_j)) / (2 h).
  ##
  ## The step @var{h} defaults to u^(1/3) max(1, |x_j|), which balances the
  ## O(h^2) truncation error against the O(u/h) rounding error of the
  ## central difference.  Returns the relative error
  ## ||g - g_fd|| / max(1, ||g||, ||g_fd||); values around 1e-8 or below
  ## indicate a correct gradient, values above ~1e-4 almost always a bug.
  ## @end deftypefn

  x = x(:);
  n = numel (x);
  [~, g] = fg (x);
  g = g(:);
  if (nargin < 3)
    h = eps ^ (1/3) * max (1, abs (x));
  else
    h = h * ones (n, 1);
  endif
  g_fd = zeros (n, 1);
  for j = 1:n
    e = zeros (n, 1);
    e(j) = h(j);
    [fp, ~] = fg (x + e);  # request both outputs: FG may be a deal() handle
    [fm, ~] = fg (x - e);
    g_fd(j) = (fp - fm) / (2 * h(j));
  endfor
  relerr = norm (g - g_fd) / max ([1, norm(g), norm(g_fd)]);
endfunction

%!test
%! fg = @(x) deal (sum (sin (x) .* x .^ 2), cos (x) .* x .^ 2 + 2 * x .* sin (x));
%! assert (omp.optim.gradcheck (fg, [0.3; -1.2; 2]) < 1e-9);

%!test
%! ## A wrong gradient is detected.
%! fg = @(x) deal (sum (x .^ 3), 2 * x .^ 2);
%! assert (omp.optim.gradcheck (fg, [1; 2]) > 1e-2);
