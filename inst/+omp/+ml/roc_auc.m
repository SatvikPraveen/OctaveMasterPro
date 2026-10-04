function [auc, se, ci, roc] = roc_auc (y, s, alpha)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{auc} =} omp.ml.roc_auc (@var{y}, @var{s})
  ## @deftypefnx {} {[@var{auc}, @var{se}, @var{ci}, @var{roc}] =} omp.ml.roc_auc (@var{y}, @var{s}, @var{alpha})
  ## Area under the ROC curve for binary labels @var{y} (0/1) and scores
  ## @var{s} (higher = more likely positive).
  ##
  ## The AUC is computed as the normalized Mann--Whitney statistic
  ## P(S+ > S-) + 0.5 P(S+ = S-) using mid-ranks, which equals the
  ## trapezoidal area under the empirical ROC curve with ties handled
  ## correctly, in O(n log n).
  ##
  ## @var{se} is the DeLong, DeLong & Clarke-Pearson (1988) nonparametric
  ## standard error, computed from the structural components (placement
  ## values).  @var{ci} is the (1 - @var{alpha}) interval (default
  ## alpha = 0.05) on the logit scale, back-transformed so it respects
  ## [0, 1].  @var{roc} has fields @code{fpr}, @code{tpr}, @code{thr}.
  ##
  ## AUC is a ranking metric: it is invariant to monotone transforms of
  ## the scores and says nothing about calibration (see
  ## @code{omp.ml.calibration}).
  ## @end deftypefn

  if (nargin < 3)
    alpha = 0.05;
  endif
  y = logical (y(:));
  s = double (s(:));
  if (numel (y) != numel (s))
    error ("omp.ml.roc_auc: Y and S must have the same length");
  endif
  n1 = sum (y);
  n0 = sum (! y);
  if (n1 == 0 || n0 == 0)
    error ("omp.ml.roc_auc: need at least one positive and one negative");
  endif

  r = midrank (s);
  auc = (sum (r(y)) - n1 * (n1 + 1) / 2) / (n1 * n0);

  if (nargout > 1)
    ## Placement values: V10_i = fraction of negatives a positive beats,
    ## V01_j = fraction of positives a negative is beaten by.
    rp = midrank (s(y));
    rn = midrank (s(! y));
    V10 = (r(y) - rp) / n0;
    V01 = 1 - (r(! y) - rn) / n1;
    S10 = var (V10);
    S01 = var (V01);
    se = sqrt (S10 / n1 + S01 / n0);
    z = -sqrt (2) * erfcinv (2 * (1 - alpha / 2));
    a = min (max (auc, 1e-12), 1 - 1e-12);
    lg = log (a / (1 - a));
    lse = se / (a * (1 - a));            # delta method
    ci = 1 ./ (1 + exp (-(lg + [-1 1] * z * lse)));
  endif

  if (nargout > 3)
    [ss, o] = sort (s, "descend");
    yy = y(o);
    keep = [diff(ss) != 0; true];        # one point per distinct threshold
    tp = cumsum (yy);
    fp = cumsum (! yy);
    roc = struct ("fpr", [0; fp(keep) / n0], "tpr", [0; tp(keep) / n1], ...
                  "thr", [Inf; ss(keep)]);
  endif
endfunction

function r = midrank (v)
  [sv, o] = sort (v);
  n = numel (v);
  r = zeros (n, 1);
  i = 1;
  while (i <= n)
    j = i;
    while (j < n && sv(j+1) == sv(i))
      j += 1;
    endwhile
    r(o(i:j)) = (i + j) / 2;
    i = j + 1;
  endwhile
endfunction

%!assert (omp.ml.roc_auc ([0 0 1 1], [0.1 0.2 0.8 0.9]), 1)
%!assert (omp.ml.roc_auc ([0 0 1 1], [0.9 0.8 0.2 0.1]), 0)
%!assert (omp.ml.roc_auc ([0 0 1 1], [0.5 0.5 0.5 0.5]), 0.5)
%!assert (omp.ml.roc_auc ([0 1 0 1], [1 2 3 4]), 0.75)

%!test
%! ## Equals brute-force pairwise definition and trapezoidal ROC area,
%! ## with heavy ties.
%! omp.repro.seed (1);
%! y = rand (300, 1) < 0.3;
%! s = round (4 * (randn (300, 1) + y));
%! [auc, ~, ~, roc] = omp.ml.roc_auc (y, s);
%! P = s(y); N = s(! y);
%! ref = mean (mean ((P > N') + 0.5 * (P == N')));
%! assert (auc, ref, 1e-12);
%! assert (trapz (roc.fpr, roc.tpr), auc, 1e-12);
%! assert (roc.fpr(end), 1); assert (roc.tpr(end), 1);

%!test
%! ## Invariance to strictly monotone score transforms.
%! omp.repro.seed (2);
%! y = rand (200, 1) < 0.5; s = randn (200, 1) + y;
%! assert (omp.ml.roc_auc (y, exp (3 * s)), omp.ml.roc_auc (y, s), 1e-14);

%!test
%! ## DeLong SE agrees with the Monte Carlo SD of the AUC estimator.
%! omp.repro.seed (3);
%! R = 400; a = zeros (R, 1); se = a;
%! for r = 1:R
%!   y = [true(60, 1); false(90, 1)];
%!   s = randn (150, 1) + 1.2 * y;
%!   [a(r), se(r)] = omp.ml.roc_auc (y, s);
%! endfor
%! assert (median (se), std (a), 0.15 * std (a));
%! ## Population AUC for a unit-variance shift d is Phi(d / sqrt 2).
%! assert (mean (a), 0.5 * erfc (-1.2 / 2), 0.01);

%!test
%! [auc, se, ci] = omp.ml.roc_auc ([0 0 0 1 1 1], [1 2 4 3 5 6]);
%! assert (ci(1) < auc && auc < ci(2));
%! assert (ci(1) >= 0 && ci(2) <= 1);

%!error <at least one positive> omp.ml.roc_auc ([1 1], [0.2 0.3])
