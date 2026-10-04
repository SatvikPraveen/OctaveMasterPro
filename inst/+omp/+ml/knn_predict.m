function p = knn_predict (Xtr, ytr, Xte, k)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{p} =} omp.ml.knn_predict (@var{Xtr}, @var{ytr}, @var{Xte}, @var{k})
  ## k-nearest-neighbour regression / class-probability estimate: the mean
  ## of @var{ytr} over the @var{k} training rows closest (Euclidean) to each
  ## row of @var{Xte}.
  ##
  ## Features are standardized with the mean and SD of @var{Xtr} only.
  ## Distances are computed in blocks to bound memory at ~32 MB.
  ##
  ## k-NN is a useful @emph{diagnostic} model in leakage studies: it can
  ## exploit near-duplicate neighbours that straddle a careless train/test
  ## split (adjacent time steps of the same unit), so the gap between its
  ## random-split and temporal-split scores measures how much a protocol
  ## rewards memorization.
  ## @end deftypefn

  if (columns (Xtr) != columns (Xte))
    error ("omp.ml.knn_predict: Xtr and Xte must have the same number of columns");
  endif
  if (any (! isfinite ([Xtr(:); Xte(:)])))
    error ("omp.ml.knn_predict: features contain NaN or Inf; impute first");
  endif
  ntr = rows (Xtr);
  k = min (k, ntr);
  mu = mean (Xtr, 1);
  sd = std (Xtr, 0, 1);
  sd(sd == 0) = 1;
  A = (Xtr - mu) ./ sd;
  B = (Xte - mu) ./ sd;
  ytr = double (ytr(:));
  na = sum (A .^ 2, 2)';
  nte = rows (B);
  p = zeros (nte, 1);
  blk = max (1, floor (4e6 / ntr));
  for i = 1:blk:nte
    j = i:min (i + blk - 1, nte);
    D = sum (B(j, :) .^ 2, 2) + na - 2 * B(j, :) * A';
    [~, o] = sort (D, 2);
    p(j) = mean (reshape (ytr(o(:, 1:k)), numel (j), k), 2);  # keep 2-D for 1 row
  endfor
endfunction

%!test
%! Xtr = [0; 1; 10; 11]; ytr = [0; 0; 1; 1];
%! assert (omp.ml.knn_predict (Xtr, ytr, [0.2; 10.4], 2), [0; 1]);
%! assert (omp.ml.knn_predict (Xtr, ytr, 5, 4), 0.5);

%!test
%! ## Agrees with a brute-force implementation.
%! omp.repro.seed (1);
%! Xtr = randn (200, 3); ytr = rand (200, 1) < 0.4; Xte = randn (50, 3);
%! p = omp.ml.knn_predict (Xtr, ytr, Xte, 7);
%! Z = (Xtr - mean (Xtr)) ./ std (Xtr); W = (Xte - mean (Xtr)) ./ std (Xtr);
%! ref = zeros (50, 1);
%! for i = 1:50
%!   [~, o] = sort (sum ((Z - W(i, :)) .^ 2, 2));
%!   ref(i) = mean (ytr(o(1:7)));
%! endfor
%! assert (p, ref, 1e-12);

%!error <impute first> omp.ml.knn_predict ([1; NaN], [0; 1], 1, 1)
