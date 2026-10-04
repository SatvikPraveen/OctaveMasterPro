function [Q, R] = gram_schmidt (A, variant)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{Q}, @var{R}] =} omp.linalg.gram_schmidt (@var{A})
  ## @deftypefnx {} {[@var{Q}, @var{R}] =} omp.linalg.gram_schmidt (@var{A}, @var{variant})
  ## Thin QR factorization of the m-by-n (m >= n) matrix @var{A} by
  ## Gram--Schmidt orthogonalization.
  ##
  ## @var{variant} selects the algorithm:
  ## @table @asis
  ## @item "mgs" (default)
  ## Modified Gram--Schmidt.  Loss of orthogonality
  ## ||I - Q'Q|| = O(u kappa(A)) (Bjorck, 1967).
  ## @item "cgs"
  ## Classical Gram--Schmidt.  Loss of orthogonality
  ## O(u kappa(A)^2) when kappa(A)^2 u < 1 (Giraud et al., 2005), and
  ## complete breakdown beyond.
  ## @item "cgs2"
  ## Classical Gram--Schmidt with one full reorthogonalization pass
  ## ("twice is enough", Giraud et al., 2005).  Orthogonality O(u) provided
  ## u kappa(A) < 1.
  ## @end table
  ##
  ## All three variants produce R with small backward error
  ## ||A - QR|| = O(u ||A||); they differ only in the orthogonality of Q.
  ## These bounds are verified empirically in this file's tests and in
  ## @file{experiments/qr_orthogonality.m}.
  ## @end deftypefn

  if (nargin < 2)
    variant = "mgs";
  endif
  [m, n] = size (A);
  if (m < n)
    error ("omp.linalg.gram_schmidt: A must have at least as many rows as columns");
  endif

  Q = zeros (m, n, class (A));
  R = zeros (n, n, class (A));
  switch (lower (variant))
    case "mgs"
      V = A;
      for k = 1:n
        R(k, k) = norm (V(:, k));
        Q(:, k) = V(:, k) / R(k, k);
        R(k, k+1:n) = Q(:, k)' * V(:, k+1:n);
        V(:, k+1:n) -= Q(:, k) * R(k, k+1:n);
      endfor

    case "cgs"
      for k = 1:n
        R(1:k-1, k) = Q(:, 1:k-1)' * A(:, k);
        v = A(:, k) - Q(:, 1:k-1) * R(1:k-1, k);
        R(k, k) = norm (v);
        Q(:, k) = v / R(k, k);
      endfor

    case "cgs2"
      for k = 1:n
        v = A(:, k);
        for pass = 1:2
          c = Q(:, 1:k-1)' * v;
          v -= Q(:, 1:k-1) * c;
          R(1:k-1, k) += c;
        endfor
        R(k, k) = norm (v);
        Q(:, k) = v / R(k, k);
      endfor

    otherwise
      error ("omp.linalg.gram_schmidt: unknown variant '%s'", variant);
  endswitch
endfunction

%!shared A
%! omp.repro.seed (11);
%! A = randn (30, 10);

%!test
%! for v = {"mgs", "cgs", "cgs2"}
%!   [Q, R] = omp.linalg.gram_schmidt (A, v{1});
%!   assert (Q * R, A, 1e-13);
%!   assert (Q' * Q, eye (10), 1e-13);
%!   assert (istriu (R));
%! endfor

%!test
%! ## Ordering of orthogonality loss on an ill-conditioned matrix:
%! ## CGS >> MGS >> CGS2, and MGS tracks u*kappa(A).
%! omp.repro.seed (5);
%! [U, ~] = qr (randn (60, 12), 0);
%! [V, ~] = qr (randn (12));
%! kappa = 1e8;
%! A = U * diag (logspace (0, -log10 (kappa), 12)) * V';
%! loss = @(Q) norm (eye (12) - Q' * Q);
%! [Qc, Rc] = omp.linalg.gram_schmidt (A, "cgs");
%! [Qm, Rm] = omp.linalg.gram_schmidt (A, "mgs");
%! [Q2, R2] = omp.linalg.gram_schmidt (A, "cgs2");
%! assert (loss (Qc) > 1e3 * loss (Qm));
%! assert (loss (Qm) > 1e3 * loss (Q2));
%! assert (loss (Qm) < 1e2 * eps * kappa);
%! assert (loss (Q2) < 1e2 * eps);
%! ## Backward error is small for every variant.
%! for QR = {{Qc, Rc}, {Qm, Rm}, {Q2, R2}}
%!   assert (norm (A - QR{1}{1} * QR{1}{2}) / norm (A) < 1e2 * eps);
%! endfor

%!error <at least as many rows> omp.linalg.gram_schmidt (ones (2, 3))
%!error <unknown variant> omp.linalg.gram_schmidt (eye (2), "foo")
