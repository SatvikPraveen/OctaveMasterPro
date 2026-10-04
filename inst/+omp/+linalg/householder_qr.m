function [Q, R] = householder_qr (A, mode)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{Q}, @var{R}] =} omp.linalg.householder_qr (@var{A})
  ## @deftypefnx {} {[@var{Q}, @var{R}] =} omp.linalg.householder_qr (@var{A}, "econ")
  ## QR factorization of the m-by-n matrix @var{A} by Householder
  ## reflections (Golub & Van Loan, @cite{Matrix Computations}, Alg. 5.2.1).
  ##
  ## The reflector for column k is chosen with the sign that avoids
  ## cancellation, v = x + sign(x_1) ||x|| e_1, which makes the algorithm
  ## backward stable: the computed factors satisfy
  ## A + dA = Q R with ||dA|| = O(u) ||A|| and ||Q'Q - I|| = O(u),
  ## independent of the condition number of @var{A} (u is the unit
  ## roundoff).  Contrast with @code{omp.linalg.gram_schmidt}.
  ##
  ## With @qcode{"econ"} (and m > n) the thin factors Q (m-by-n) and
  ## R (n-by-n) are returned.  The diagonal of @var{R} is made non-negative
  ## so the factorization is unique for full-rank @var{A}.
  ## @end deftypefn

  if (nargin < 2)
    mode = "full";
  endif
  if (! (ismatrix (A) && isnumeric (A) && ndims (A) == 2))
    error ("omp.linalg.householder_qr: A must be a 2-D numeric matrix");
  endif
  econ = strcmpi (mode, "econ") || isequal (mode, 0);

  [m, n] = size (A);
  R = A;
  p = min (m - 1, n);
  V = cell (1, p);
  for k = 1:p
    x = R(k:m, k);
    nx = norm (x);
    v = x;
    if (nx == 0)
      V{k} = [];
      continue;
    endif
    s = sign (x(1));
    if (s == 0)
      s = 1;
    endif
    v(1) += s * nx;
    v /= norm (v);
    V{k} = v;
    R(k:m, k:n) -= 2 * v * (v' * R(k:m, k:n));
  endfor

  ## Accumulate Q by backward application of the reflectors (cheaper than
  ## forming each H_k explicitly).
  if (econ && m > n)
    Q = eye (m, n);
  else
    Q = eye (m);
  endif
  for k = p:-1:1
    v = V{k};
    if (isempty (v))
      continue;
    endif
    Q(k:m, :) -= 2 * v * (v' * Q(k:m, :));
  endfor

  if (econ && m > n)
    R = R(1:n, :);
  endif
  R = triu (R);

  ## Normalize so that diag(R) >= 0.
  r = min (size (R));
  d = sign (diag (R(1:r, 1:r)));  # diag() of a row vector would build a matrix
  d(d == 0) = 1;
  Q(:, 1:r) = Q(:, 1:r) .* d';
  R(1:r, :) = d .* R(1:r, :);
endfunction

%!test
%! omp.repro.seed (1);
%! A = randn (8, 5);
%! [Q, R] = omp.linalg.householder_qr (A);
%! assert (size (Q), [8 8]);
%! assert (size (R), [8 5]);
%! assert (Q * R, A, 10 * eps * norm (A));
%! assert (Q' * Q, eye (8), 10 * eps);
%! assert (istriu (R));
%! assert (all (diag (R) >= 0));

%!test
%! omp.repro.seed (2);
%! A = randn (10, 4);
%! [Q, R] = omp.linalg.householder_qr (A, "econ");
%! assert (size (Q), [10 4]);
%! assert (size (R), [4 4]);
%! ## Uniqueness: agrees with LAPACK after the same sign normalization.
%! [Q0, R0] = qr (A, 0);
%! d = sign (diag (R0));
%! assert (R, d .* R0, 1e-12);
%! assert (Q, Q0 .* d', 1e-12);

%!test
%! ## Wide and square matrices.
%! omp.repro.seed (3);
%! for sz = {[3 6], [5 5], [1 4], [4 1]}
%!   A = randn (sz{1});
%!   [Q, R] = omp.linalg.householder_qr (A);
%!   assert (Q * R, A, 1e-13);
%! endfor

%!test
%! ## Orthogonality is independent of conditioning (backward stability).
%! A = hilb (12);                      # cond ~ 1.7e16
%! [Q, R] = omp.linalg.householder_qr (A, "econ");
%! assert (norm (Q' * Q - eye (12)) < 50 * eps);
%! assert (norm (A - Q * R) / norm (A) < 50 * eps);

%!test
%! ## Rank-deficient input with a zero column.
%! A = [1 0 2; 2 0 4; 3 0 6];
%! [Q, R] = omp.linalg.householder_qr (A);
%! assert (Q * R, A, 1e-14);

%!error <2-D numeric> omp.linalg.householder_qr ({1})
