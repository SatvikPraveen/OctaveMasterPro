function c = calibration (y, p, nbins)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{c} =} omp.ml.calibration (@var{y}, @var{p})
  ## @deftypefnx {} {@var{c} =} omp.ml.calibration (@var{y}, @var{p}, @var{nbins})
  ## Probability calibration diagnostics for binary outcomes @var{y} and
  ## predicted probabilities @var{p}.
  ##
  ## Returns a struct with
  ## @table @code
  ## @item brier
  ## Brier score mean((p - y)^2) (lower is better).
  ## @item brier_ref
  ## Brier score of the constant prevalence forecast, pi (1 - pi).
  ## @item brier_skill
  ## 1 - brier / brier_ref; positive means better than predicting the base
  ## rate.  The appropriate zero-information baseline for rare events.
  ## @item log_loss
  ## Mean negative log-likelihood (probabilities clipped to [1e-15, 1 - 1e-15]).
  ## @item ece
  ## Expected calibration error with @var{nbins} (default 10) equal-mass
  ## bins: sum_b (n_b / n) |mean(y_b) - mean(p_b)|.  Equal-mass bins avoid
  ## the empty or near-empty high-probability bins of equal-width binning.
  ## @item table
  ## Reliability-diagram data: [mean_p, mean_y, count] per bin.
  ## @end table
  ##
  ## ECE is a biased estimator (positive even for perfect calibration at
  ## finite n); compare against its value for the same n under a
  ## perfectly calibrated simulation when interpreting small differences.
  ## @end deftypefn

  if (nargin < 3)
    nbins = 10;
  endif
  y = double (y(:));
  p = double (p(:));
  if (numel (y) != numel (p))
    error ("omp.ml.calibration: Y and P must have the same length");
  endif
  if (any (p < 0 | p > 1))
    error ("omp.ml.calibration: P must lie in [0, 1]");
  endif
  n = numel (y);
  pi_ = mean (y);

  c.brier = mean ((p - y) .^ 2);
  c.brier_ref = pi_ * (1 - pi_);
  c.brier_skill = 1 - c.brier / c.brier_ref;
  pc = min (max (p, 1e-15), 1 - 1e-15);
  c.log_loss = -mean (y .* log (pc) + (1 - y) .* log (1 - pc));

  [~, o] = sort (p);
  edges = round (linspace (0, n, nbins + 1));
  T = zeros (nbins, 3);
  for b = 1:nbins
    k = o(edges(b)+1:edges(b+1));
    if (isempty (k))
      T(b, :) = [NaN NaN 0];
    else
      T(b, :) = [mean(p(k)), mean(y(k)), numel(k)];
    endif
  endfor
  T = T(T(:, 3) > 0, :);
  c.ece = sum (T(:, 3) / n .* abs (T(:, 2) - T(:, 1)));
  c.table = T;
endfunction

%!test
%! ## Perfectly calibrated forecasts: small ECE, positive skill.
%! omp.repro.seed (1);
%! p = rand (50000, 1); y = rand (50000, 1) < p;
%! c = omp.ml.calibration (y, p);
%! assert (c.ece < 0.01);
%! assert (c.brier, 1/6, 0.005);            # E[p(1-p)] for p ~ U(0,1)
%! assert (c.brier_skill > 0.3);

%!test
%! ## Systematic over-confidence is detected.
%! omp.repro.seed (2);
%! p = rand (50000, 1); y = rand (50000, 1) < p;
%! q = min (1, max (0, 0.5 + 1.8 * (p - 0.5)));
%! c0 = omp.ml.calibration (y, p);
%! c1 = omp.ml.calibration (y, q);
%! assert (c1.ece > 5 * c0.ece);
%! assert (c1.log_loss > c0.log_loss);

%!test
%! ## Constant prevalence forecast has zero Brier skill.
%! y = [1; 0; 0; 0];
%! c = omp.ml.calibration (y, 0.25 * ones (4, 1), 2);
%! assert (c.brier_skill, 0, 1e-15);
%! assert (sum (c.table(:, 3)), 4);

%!error <\[0, 1\]> omp.ml.calibration ([0 1], [0.5 1.5])
