function q = p_adjust (p, method)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{q} =} omp.stats.p_adjust (@var{p}, @var{method})
  ## Adjust p-values for multiple comparisons.  Returns adjusted values of
  ## the same shape as @var{p}; NaN entries are ignored and returned as NaN.
  ##
  ## @table @asis
  ## @item "bonferroni"
  ## Family-wise error rate (FWER) control under arbitrary dependence.
  ## @item "holm"
  ## Step-down FWER control (Holm, 1979); uniformly more powerful than
  ## Bonferroni under the same assumptions.
  ## @item "bh"
  ## False discovery rate (FDR) control (Benjamini & Hochberg, 1995);
  ## valid under independence or positive regression dependence.
  ## @item "by"
  ## FDR control under arbitrary dependence (Benjamini & Yekutieli, 2001).
  ## @end table
  ##
  ## Results agree with R's @code{p.adjust}.
  ## @end deftypefn

  if (nargin < 2)
    method = "holm";
  endif
  sz = size (p);
  p = p(:);
  ok = ! isnan (p);
  pv = p(ok);
  if (any (pv < 0 | pv > 1))
    error ("omp.stats.p_adjust: p-values must lie in [0, 1]");
  endif
  m = numel (pv);
  [ps, o] = sort (pv);
  i = (1:m)';

  switch (lower (method))
    case "bonferroni"
      qs = min (1, m * ps);
    case "holm"
      qs = min (1, cummax ((m - i + 1) .* ps));
    case "bh"
      qs = min (1, flipud (cummin (flipud (m ./ i .* ps))));
    case "by"
      c = sum (1 ./ (1:m));
      qs = min (1, flipud (cummin (flipud (c * m ./ i .* ps))));
    otherwise
      error ("omp.stats.p_adjust: unknown method '%s'", method);
  endswitch

  out = zeros (m, 1);
  out(o) = qs;
  q = NaN (size (p));
  q(ok) = out;
  q = reshape (q, sz);
endfunction

%!shared p
%! p = [0.01 0.02 0.03 0.04 0.05];

## Reference values computed with R 4.x p.adjust().
%!assert (omp.stats.p_adjust (p, "bonferroni"), [0.05 0.10 0.15 0.20 0.25], 1e-12)
%!assert (omp.stats.p_adjust (p, "holm"), [0.05 0.08 0.09 0.09 0.09], 1e-12)
%!assert (omp.stats.p_adjust (p, "bh"), [0.05 0.05 0.05 0.05 0.05], 1e-12)
%!assert (omp.stats.p_adjust (p, "by"), 0.05 * 137/60 * ones (1, 5), 1e-12)

%!test
%! ## Order invariance, NaN handling, shape preservation.
%! q = omp.stats.p_adjust ([0.04; NaN; 0.001; 0.3], "bh");
%! assert (size (q), [4 1]);
%! assert (isnan (q(2)));
%! assert (q([1 3 4]), [0.06; 0.003; 0.3], 1e-12);

%!test
%! ## Monotone in the raw p-values, and BH <= Holm <= Bonferroni.
%! omp.repro.seed (1);
%! r = rand (1, 50) .^ 3;
%! b = omp.stats.p_adjust (r, "bonferroni");
%! h = omp.stats.p_adjust (r, "holm");
%! f = omp.stats.p_adjust (r, "bh");
%! assert (all (f <= h + 1e-15) && all (h <= b + 1e-15));
%! [~, o] = sort (r);
%! assert (all (diff (f(o)) >= 0));

%!error <unknown method> omp.stats.p_adjust (0.1, "foo")
%!error <\[0, 1\]> omp.stats.p_adjust (1.5, "bh")
