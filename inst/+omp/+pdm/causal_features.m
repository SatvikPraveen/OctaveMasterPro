function [X, names] = causal_features (d, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{X}, @var{names}] =} omp.pdm.causal_features (@var{d})
  ## @deftypefnx {} {[@var{X}, @var{names}] =} omp.pdm.causal_features (@var{d}, "Short", @var{ws}, "Slope", @var{wl}, "Centered", @var{tf})
  ## Rolling-window features for predictive maintenance computed, by
  ## default, from @emph{past and present observations only}.
  ##
  ## @var{d} is a struct of equal-length column vectors with fields
  ## @code{unit}, @code{t} (integer hours, sorted within unit),
  ## @code{ambient}, @code{temperature}, @code{vibration}, @code{pressure}
  ## and, optionally, @code{failure} and @code{maintenance} event flags
  ## (as from @code{omp.pdm.simulate_fleet}).
  ##
  ## Three derived signals are used: temperature minus ambient (removes
  ## the shared confounder), log vibration, and pressure.  For each signal
  ## the features are
  ## @itemize
  ## @item @code{mean_s}: mean over the window (t - ws, t], default ws = 24 h;
  ## @item @code{std_s}: standard deviation over the same window;
  ## @item @code{slope_l}: least-squares slope over (t - wl, t], wl = 72 h;
  ## @item @code{dev}: mean_s minus the unit's expanding mean of all
  ## readings up to t (removes unit-specific offsets causally).
  ## @end itemize
  ## plus @code{hours_since_renewal}, the time since the last failure or
  ## maintenance event (known to an operator in real time).
  ##
  ## Missing readings (NaN) are skipped: every statistic uses the
  ## available readings in its window, and is NaN if there are too few.
  ##
  ## @var{Centered} = true deliberately shifts every window to be centred
  ## on t, so features use up to ws/2 (wl/2) hours of @emph{future} data.
  ## This exists only to measure the optimism caused by such look-ahead
  ## leakage; never use it for real evaluation.
  ## @end deftypefn

  opts = struct ("short", 24, "slope", 72, "centered", false);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.pdm.causal_features: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor

  sig = {(d.temperature - d.ambient), log(d.vibration), d.pressure};  # no space before ( inside {}
  sname = {"temp_resid", "log_vib", "pressure"};
  n = numel (d.t);
  names = {};
  for j = 1:3
    names = [names, strcat(sname{j}, {"_mean", "_std", "_slope", "_dev"})];
  endfor
  names{end+1} = "hours_since_renewal";
  X = NaN (n, numel (names));

  renew = false (n, 1);
  if (isfield (d, "failure"))
    renew |= logical (d.failure);
  endif
  if (isfield (d, "maintenance"))
    renew |= logical (d.maintenance);
  endif

  units = unique (d.unit)';
  for u = units
    r = find (d.unit == u);
    tt = d.t(r);
    if (any (diff (tt) <= 0))
      error ("omp.pdm.causal_features: times must be strictly increasing within a unit");
    endif
    for j = 1:3
      x = sig{j}(r);
      [m, s] = window_moments (tt, x, opts.short, opts.centered);
      b = window_slope (tt, x, opts.slope, opts.centered);
      ok = ! isnan (x);
      cs = cumsum (ifelse (ok, x, 0));
      cn = cumsum (ok);
      expmean = cs ./ cn;
      expmean(cn == 0) = NaN;
      X(r, (j - 1) * 4 + (1:4)) = [m, s, b, m - expmean];
    endfor
    ## Hours since the last renewal event at or before t.
    last = NaN (numel (r), 1);
    cur = tt(1);                    # unknown history: count from start
    ev = renew(r);
    for i = 1:numel (r)
      if (ev(i))
        cur = tt(i);
      endif
      last(i) = tt(i) - cur;
    endfor
    X(r, end) = last;
  endfor
endfunction

function v = ifelse (c, a, b)
  v = b * ones (size (c));
  v(c) = a(c);
endfunction

function [lo, hi] = window_bounds (t, w, centered)
  ## Window (t - w, t] or (t - w/2, t + w/2], returned as index ranges into
  ## the cumulative sums: elements lo+1 .. hi.
  if (centered)
    a = t - w / 2;
    b = t + w / 2;
  else
    a = t - w;
    b = t;
  endif
  lo = lookup (t, a);               # number of samples with time <= a
  hi = lookup (t, b);
endfunction

function [m, s] = window_moments (t, x, w, centered)
  ok = ! isnan (x);
  x0 = x;
  x0(! ok) = 0;
  C1 = [0; cumsum(x0)];
  C2 = [0; cumsum(x0 .^ 2)];
  Cn = [0; cumsum(ok)];
  [lo, hi] = window_bounds (t, w, centered);
  n = Cn(hi + 1) - Cn(lo + 1);
  s1 = C1(hi + 1) - C1(lo + 1);
  s2 = C2(hi + 1) - C2(lo + 1);
  m = s1 ./ n;
  v = (s2 - s1 .^ 2 ./ n) ./ (n - 1);
  s = sqrt (max (v, 0));
  m(n < 3) = NaN;
  s(n < 3) = NaN;
endfunction

function b = window_slope (t, x, w, centered)
  ok = ! isnan (x);
  tc = t - t(1);                    # improves conditioning of the sums
  x0 = x; x0(! ok) = 0;
  t0 = tc; t0(! ok) = 0;
  S = @(v) [0; cumsum(v)];
  Cn = S (double (ok)); Ct = S (t0); Cx = S (x0);
  Ctt = S (t0 .^ 2); Ctx = S (t0 .* x0);
  [lo, hi] = window_bounds (t, w, centered);
  g = @(C) C(hi + 1) - C(lo + 1);
  n = g (Cn); st = g (Ct); sx = g (Cx); stt = g (Ctt); stx = g (Ctx);
  den = n .* stt - st .^ 2;
  b = (n .* stx - st .* sx) ./ den;
  b(n < 3 | den <= 0) = NaN;
endfunction

%!shared s
%! s = omp.pdm.simulate_fleet ("Units", 4, "Hours", 24 * 15, "Seed", 2);

%!test
%! [X, names] = omp.pdm.causal_features (s);
%! assert (size (X), [numel(s.t), 13]);
%! assert (numel (names), 13);
%! ## Window mean equals a direct computation at an arbitrary row.
%! r = find (s.unit == 2 & s.t == 200);
%! w = s.unit == 2 & s.t > 176 & s.t <= 200;
%! v = log (s.vibration(w));
%! assert (X(r, 5), mean (v(! isnan (v))), 1e-12);
%! assert (X(r, 6), std (v(! isnan (v))), 1e-12);
%! ## Slope matches polyfit over the 72 h window.
%! w = s.unit == 2 & s.t > 128 & s.t <= 200 & ! isnan (s.pressure);
%! pf = polyfit (s.t(w), s.pressure(w), 1);
%! assert (X(r, 11), pf(1), 1e-10);

%!test
%! ## CAUSALITY: perturbing the future must not change features at or
%! ## before the cut time; the centred variant must change (detects leak).
%! cut = 150;
%! s2 = s;
%! fut = s.t > cut;
%! s2.vibration(fut) *= 5;
%! s2.temperature(fut) += 20;
%! s2.pressure(fut) -= 10;
%! past = s.t <= cut;
%! A = omp.pdm.causal_features (s);
%! B = omp.pdm.causal_features (s2);
%! assert (isequaln (A(past, :), B(past, :)));
%! Ac = omp.pdm.causal_features (s, "Centered", true);
%! Bc = omp.pdm.causal_features (s2, "Centered", true);
%! assert (! isequaln (Ac(past, :), Bc(past, :)));

%!test
%! ## Units are processed independently.
%! A = omp.pdm.causal_features (s);
%! s3 = s; k = s.unit == 4; s3.vibration(k) *= 3;
%! B = omp.pdm.causal_features (s3);
%! assert (isequaln (A(! k, :), B(! k, :)));
