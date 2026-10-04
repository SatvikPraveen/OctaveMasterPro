function r = timeit (fn, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{r} =} omp.bench.timeit (@var{fn})
  ## @deftypefnx {} {@var{r} =} omp.bench.timeit (@var{fn}, "Repeats", @var{n}, "Warmup", @var{w}, "MinTime", @var{t})
  ## Robustly time the zero-argument function handle @var{fn}.
  ##
  ## The handle is first called @var{w} times (default 2) to absorb JIT,
  ## page-fault and cache warm-up effects.  It is then called @var{n} times
  ## (default 15).  If a single call is shorter than @var{t} seconds
  ## (default 1e-3), calls are batched so each sample lasts at least
  ## @var{t}, reducing timer-resolution bias.
  ##
  ## Returns a struct with fields @code{median}, @code{iqr}, @code{mean},
  ## @code{std}, @code{min}, @code{samples} (seconds per call) and
  ## @code{batch} (calls per sample).  The median and IQR are the
  ## recommended summaries: wall-clock timings are right-skewed, so the
  ## mean is dominated by outliers from OS scheduling.
  ## @end deftypefn

  opts = struct ("repeats", 15, "warmup", 2, "mintime", 1e-3);
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.bench.timeit: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor
  if (! is_function_handle (fn))
    error ("omp.bench.timeit: FN must be a function handle");
  endif

  for k = 1:opts.warmup
    fn ();
  endfor

  ## Calibrate batch size.
  batch = 1;
  t = run_batch (fn, batch);
  while (t < opts.mintime && batch < 2^20)
    batch *= max (2, ceil (opts.mintime / max (t, eps) / batch));
    t = run_batch (fn, batch);
  endwhile

  s = zeros (opts.repeats, 1);
  for k = 1:opts.repeats
    s(k) = run_batch (fn, batch) / batch;
  endfor

  q = quantiles (s, [0.25 0.5 0.75]);
  r = struct ("median", q(2), "iqr", q(3) - q(1), "mean", mean (s), ...
              "std", std (s), "min", min (s), "samples", s, "batch", batch);
endfunction

function t = run_batch (fn, n)
  t0 = tic ();
  for i = 1:n
    fn ();
  endfor
  t = toc (t0);
endfunction

function q = quantiles (x, p)
  ## Type-7 (linear interpolation) quantiles, matching R's default.
  x = sort (x(:));
  n = numel (x);
  h = (n - 1) * p + 1;
  lo = floor (h);
  hi = min (lo + 1, n);
  q = x(lo)' + (h - lo) .* (x(hi)' - x(lo)');
endfunction

%!test
%! r = omp.bench.timeit (@() sum (rand (100, 1)), "Repeats", 5);
%! assert (isfield (r, {"median", "iqr", "samples", "batch"}));
%! assert (numel (r.samples), 5);
%! assert (r.median > 0);
%! assert (r.min <= r.median);

%!test
%! ## A slower function must time slower.
%! a = omp.bench.timeit (@() pinv (rand (5)), "Repeats", 5);
%! b = omp.bench.timeit (@() pinv (rand (150)), "Repeats", 5);
%! assert (b.median > a.median);

%!error <function handle> omp.bench.timeit (1)
