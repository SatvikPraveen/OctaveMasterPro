function folds = purged_time_splits (t, k, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{folds} =} omp.ml.purged_time_splits (@var{t}, @var{k})
  ## @deftypefnx {} {@var{folds} =} omp.ml.purged_time_splits (@var{t}, @var{k}, "Horizon", @var{h}, "Embargo", @var{e}, "MinTrain", @var{f})
  ## Forward-chaining (expanding-window) train/test splits for
  ## time-indexed samples, with purging and an embargo to prevent
  ## look-ahead leakage.
  ##
  ## @var{t} gives the time stamp of each sample (any numeric unit, e.g.
  ## hours or datenum days).  The time range after an initial training
  ## fraction @var{f} (default 0.4) is cut into @var{k} contiguous test
  ## blocks.  For each block [t_a, t_b]:
  ## @itemize
  ## @item the training set contains only samples with t < t_a;
  ## @item @emph{purging}: if each label looks @var{h} time units into the
  ## future (e.g. "fails within h hours"), training samples with
  ## t + h >= t_a are removed, because their labels are determined by
  ## events inside the test period (Lopez de Prado, 2018, Ch. 7);
  ## @item @emph{embargo}: an additional gap @var{e} is removed before
  ## t_a to absorb serial correlation in the features.
  ## @end itemize
  ##
  ## Returns a struct array with logical masks @code{train} and @code{test}
  ## and the test window bounds @code{t_start}, @code{t_end}.
  ##
  ## Random K-fold CV on such data typically inflates performance because
  ## neighbouring, highly correlated samples land on both sides of the
  ## split; compare both in your experiments to quantify the effect.
  ## @end deftypefn

  opts = struct ("horizon", 0, "embargo", 0, "mintrain", 0.4);
  for j = 1:2:numel (varargin)
    key = lower (varargin{j});
    if (! isfield (opts, key))
      error ("omp.ml.purged_time_splits: unknown option '%s'", varargin{j});
    endif
    opts.(key) = varargin{j+1};
  endfor
  t = t(:);
  if (! (isscalar (k) && k >= 1 && k == fix (k)))
    error ("omp.ml.purged_time_splits: K must be a positive integer");
  endif

  t0 = min (t);
  t1 = max (t);
  start = t0 + opts.mintrain * (t1 - t0);
  bounds = linspace (start, t1, k + 1);
  folds = struct ("train", {}, "test", {}, "t_start", {}, "t_end", {});
  for i = 1:k
    a = bounds(i);
    b = bounds(i+1);
    if (i < k)
      test = t >= a & t < b;
    else
      test = t >= a & t <= b;
    endif
    train = t < a & (t + opts.horizon) < a & t < a - opts.embargo;
    if (! any (test) || ! any (train))
      continue;
    endif
    folds(end+1) = struct ("train", train, "test", test, "t_start", a, ...
                           "t_end", b);
  endfor
  if (isempty (folds))
    error ("omp.ml.purged_time_splits: no non-empty folds; reduce K, Horizon or Embargo");
  endif
endfunction

%!test
%! t = (1:100)';
%! F = omp.ml.purged_time_splits (t, 4, "Horizon", 5, "Embargo", 2);
%! assert (numel (F), 4);
%! for i = 1:numel (F)
%!   tr = t(F(i).train); te = t(F(i).test);
%!   ## No training label window reaches into the test period.
%!   assert (max (tr) + 5 < min (te));
%!   assert (max (tr) < min (te) - 2);
%! endfor
%! ## Test blocks are disjoint and cover the evaluation period.
%! cover = sum ([F.test], 2);
%! assert (all (cover <= 1));
%! assert (all (cover(t >= 1 + 0.4 * 99) == 1));

%!test
%! ## Unsorted and repeated time stamps (multiple units per time step).
%! t = repmat ((1:50)', 3, 1);
%! t = t(randperm (numel (t)));
%! F = omp.ml.purged_time_splits (t, 3, "Horizon", 2);
%! for i = 1:numel (F)
%!   assert (! any (F(i).train & F(i).test));
%!   assert (max (t(F(i).train)) + 2 < min (t(F(i).test)));
%! endfor

%!error <no non-empty folds> omp.ml.purged_time_splits (1:10, 2, "Horizon", 100)
