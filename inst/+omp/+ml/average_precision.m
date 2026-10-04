function [ap, pr] = average_precision (y, s)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{ap} =} omp.ml.average_precision (@var{y}, @var{s})
  ## @deftypefnx {} {[@var{ap}, @var{pr}] =} omp.ml.average_precision (@var{y}, @var{s})
  ## Average precision (area under the precision--recall curve, step-wise
  ## interpolation): AP = sum_k (R_k - R_(k-1)) P_k over distinct score
  ## thresholds, as in scikit-learn's @code{average_precision_score}.
  ##
  ## Unlike ROC AUC, AP depends on the class prevalence pi: a random
  ## scorer has expected AP ~ pi, so always report pi alongside AP for
  ## rare-event problems such as failure prediction.
  ##
  ## @var{pr} has fields @code{precision}, @code{recall}, @code{thr}.
  ## @end deftypefn

  y = logical (y(:));
  s = double (s(:));
  n1 = sum (y);
  if (n1 == 0)
    error ("omp.ml.average_precision: need at least one positive");
  endif
  [ss, o] = sort (s, "descend");
  yy = y(o);
  keep = [diff(ss) != 0; true];
  tp = cumsum (yy)(keep);
  k = find (keep);
  precision = tp ./ k;
  recall = tp / n1;
  ap = sum (diff ([0; recall]) .* precision);
  pr = struct ("precision", precision, "recall", recall, "thr", ss(keep));
endfunction

%!assert (omp.ml.average_precision ([1 1 0 0], [4 3 2 1]), 1)
%!assert (omp.ml.average_precision ([0 0 1 1], [4 3 2 1]), (0.5 * 1/3 + 0.5 * 2/4), 1e-15)

%!test
%! ## Reference value from scikit-learn's documentation example:
%! ## average_precision_score([0,0,1,1], [0.1,0.4,0.35,0.8]) = 0.8333...
%! assert (omp.ml.average_precision ([0 0 1 1], [0.1 0.4 0.35 0.8]), 5/6, 1e-12);

%!test
%! ## Random scores: AP ~ prevalence.
%! omp.repro.seed (1);
%! y = rand (20000, 1) < 0.05;
%! assert (omp.ml.average_precision (y, rand (20000, 1)), mean (y), 0.01);

%!test
%! ## Ties are treated as a single threshold.
%! assert (omp.ml.average_precision ([1 0 1 0], [1 1 1 1]), 0.5);
