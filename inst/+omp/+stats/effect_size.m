function es = effect_size (x, y, type)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{es} =} omp.stats.effect_size (@var{x}, @var{y}, @var{type})
  ## Standardized effect size of the difference between samples @var{x}
  ## and @var{y}.
  ##
  ## @table @asis
  ## @item "hedges_g" (default)
  ## Cohen's d with pooled SD, multiplied by the small-sample bias
  ## correction J = 1 - 3 / (4 (n_x + n_y) - 9) (Hedges, 1981).
  ## @item "cohens_d"
  ## (mean(x) - mean(y)) / s_pooled.
  ## @item "cliffs_delta"
  ## P(X > Y) - P(X < Y), a rank-based effect in [-1, 1] that is
  ## invariant to monotone transforms and robust to outliers
  ## (Cliff, 1993).  Equals 2 AUC - 1 when x are scores of positives and
  ## y of negatives.
  ## @end table
  ##
  ## Report an effect size with every significance test: with large n a
  ## negligible effect can be highly "significant".
  ## @end deftypefn

  if (nargin < 3)
    type = "hedges_g";
  endif
  x = x(:);
  y = y(:);
  nx = numel (x);
  ny = numel (y);
  switch (lower (type))
    case {"cohens_d", "hedges_g"}
      if (nx < 2 || ny < 2)
        error ("omp.stats.effect_size: need >= 2 observations per group");
      endif
      sp = sqrt (((nx - 1) * var (x) + (ny - 1) * var (y)) / (nx + ny - 2));
      es = (mean (x) - mean (y)) / sp;
      if (strcmpi (type, "hedges_g"))
        es *= 1 - 3 / (4 * (nx + ny) - 9);
      endif
    case "cliffs_delta"
      ## O((nx + ny) log(nx + ny)) via ranks instead of the O(nx ny) grid.
      r = midrank ([x; y]);
      u = sum (r(1:nx)) - nx * (nx + 1) / 2;   # Mann-Whitney U for x
      es = 2 * u / (nx * ny) - 1;
    otherwise
      error ("omp.stats.effect_size: unknown type '%s'", type);
  endswitch
endfunction

function r = midrank (v)
  ## Mid-ranks (ties share the average of their ranks), vectorized.
  [sv, o] = sort (v(:));
  n = numel (sv);
  starts = [true; diff(sv) != 0];
  grp = cumsum (starts);
  first = find (starts);
  last = [first(2:end) - 1; n];
  mid = (first + last) / 2;
  r = zeros (n, 1);
  r(o) = mid(grp);
endfunction

%!test
%! x = [2 4 6 8]; y = [1 3 5 7];
%! sp = sqrt ((3 * var (x) + 3 * var (y)) / 6);
%! assert (omp.stats.effect_size (x, y, "cohens_d"), 1 / sp, 1e-12);
%! assert (omp.stats.effect_size (x, y), (1 / sp) * (1 - 3 / 23), 1e-12);

%!test
%! ## Cliff's delta: extremes, ties and agreement with the O(n^2) definition.
%! assert (omp.stats.effect_size (5:9, 1:4, "cliffs_delta"), 1);
%! assert (omp.stats.effect_size (1:4, 5:9, "cliffs_delta"), -1);
%! assert (omp.stats.effect_size ([1 1 1], [1 1], "cliffs_delta"), 0);
%! omp.repro.seed (1);
%! x = round (3 * randn (40, 1)); y = round (3 * randn (35, 1) + 1);
%! ref = mean (mean (sign (x - y')));
%! assert (omp.stats.effect_size (x, y, "cliffs_delta"), ref, 1e-12);

%!test
%! ## Hedges' g is approximately unbiased where Cohen's d is not (n = 5).
%! omp.repro.seed (2);
%! R = 4000; d = zeros (R, 1); g = d;
%! for r = 1:R
%!   a = randn (5, 1) + 1; b = randn (5, 1);
%!   d(r) = omp.stats.effect_size (a, b, "cohens_d");
%!   g(r) = omp.stats.effect_size (a, b, "hedges_g");
%! endfor
%! assert (abs (mean (g) - 1) < abs (mean (d) - 1));
%! assert (abs (mean (g) - 1) < 0.05);

%!error <unknown type> omp.stats.effect_size (1:3, 1:3, "foo")
