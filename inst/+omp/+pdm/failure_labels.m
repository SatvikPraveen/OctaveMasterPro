function [y, eligible, ttf] = failure_labels (d, horizon)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{y}, @var{eligible}, @var{ttf}] =} omp.pdm.failure_labels (@var{d}, @var{horizon})
  ## Binary targets "the unit fails within the next @var{horizon} hours":
  ## y(i) = 1 iff the same unit has a failure at a time in
  ## (t_i, t_i + horizon].
  ##
  ## @var{eligible} marks rows that may be used for training or evaluation:
  ## the unit is online, the row is not itself a failure hour, and the
  ## label is observable -- rows whose horizon extends past the unit's
  ## last observation without an observed failure are right-censored and
  ## excluded rather than silently labelled negative.
  ##
  ## @var{ttf} is the time to the next failure (Inf if none observed).
  ## @end deftypefn

  if (! (isscalar (horizon) && horizon > 0))
    error ("omp.pdm.failure_labels: HORIZON must be a positive scalar");
  endif
  n = numel (d.t);
  ttf = Inf (n, 1);
  units = unique (d.unit)';
  tend = zeros (n, 1);
  for u = units
    r = find (d.unit == u);
    tt = d.t(r);
    f = logical (d.failure(r));
    nxt = Inf;
    for i = numel (r):-1:1
      ttf(r(i)) = nxt - tt(i);
      if (f(i))
        nxt = tt(i);
      endif
    endfor
    tend(r) = tt(end);
  endfor
  y = ttf <= horizon;
  censored = ! y & (d.t + horizon > tend);
  online = true (n, 1);
  if (isfield (d, "online"))
    online = logical (d.online);
  endif
  eligible = online & ! logical (d.failure) & ! censored;
endfunction

%!test
%! d.unit = [1 1 1 1 1 1 2 2 2]';
%! d.t = [0 1 2 3 4 5 0 1 2]';
%! d.failure = logical ([0 0 0 1 0 0 0 0 0]');
%! [y, e, ttf] = omp.pdm.failure_labels (d, 2);
%! assert (ttf(1:4), [3; 2; 1; Inf]);
%! assert (y', logical ([0 1 1 0 0 0 0 0 0]));
%! ## Row 4 is the failure itself; rows 5-6 and 8-9 are censored.
%! assert (e', logical ([1 1 1 0 0 0 1 0 0]));

%!test
%! s = omp.pdm.simulate_fleet ("Units", 5, "Seed", 3);
%! [y, e] = omp.pdm.failure_labels (s, 72);
%! assert (! any (e & ! s.online));
%! ## Every failure is preceded by positive eligible labels.
%! k = find (s.failure);
%! assert (all (y(max (k - 1, 1)) | s.unit(max (k - 1, 1)) != s.unit(k) | ! s.online(max (k - 1, 1))));

%!error <positive scalar> omp.pdm.failure_labels (struct ("t", 1), 0)
