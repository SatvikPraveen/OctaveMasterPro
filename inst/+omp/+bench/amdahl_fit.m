function fit = amdahl_fit (p, speedup)
  ## -*- texinfo -*-
  ## @deftypefn {} {@var{fit} =} omp.bench.amdahl_fit (@var{p}, @var{speedup})
  ## Fit Amdahl's law
  ## @tex
  ## $$ S(p) = {1 \over s + (1-s)/p} $$
  ## @end tex
  ## @ifnottex
  ## S(p) = 1 / (s + (1 - s)/p)
  ## @end ifnottex
  ## to measured speedups @var{speedup} at worker counts @var{p}, estimating
  ## the serial fraction @var{s} in [0, 1].
  ##
  ## Because 1/S = s (1 - 1/p) + 1/p is linear in @var{s}, the fit is a
  ## closed-form one-parameter least squares on the reciprocal speedups,
  ## clipped to [0, 1].  The returned struct contains @code{serial_fraction},
  ## @code{max_speedup} (= 1/s, the asymptotic ceiling), @code{predicted}
  ## speedups and the @code{rmse} of the fit in speedup units.
  ##
  ## A residual pattern that grows with @var{p} (speedup falling below the
  ## fit) indicates overheads Amdahl's law does not model: communication,
  ## memory bandwidth saturation or oversubscription.
  ## @end deftypefn

  p = p(:);
  speedup = speedup(:);
  if (numel (p) != numel (speedup) || numel (p) < 2)
    error ("omp.bench.amdahl_fit: P and SPEEDUP must have equal length >= 2");
  endif
  if (any (p < 1) || any (speedup <= 0))
    error ("omp.bench.amdahl_fit: P must be >= 1 and SPEEDUP > 0");
  endif

  a = 1 - 1 ./ p;          # regressor
  b = 1 ./ speedup - 1 ./ p;
  if (all (a == 0))
    s = NaN;
  else
    s = (a' * b) / (a' * a);
    s = min (max (s, 0), 1);
  endif

  pred = 1 ./ (s + (1 - s) ./ p);
  fit = struct ("serial_fraction", s, "max_speedup", 1 / s, ...
                "predicted", pred, "residuals", speedup - pred, ...
                "rmse", sqrt (mean ((speedup - pred) .^ 2)));
endfunction

%!test
%! p = [1 2 4 8 16]';
%! s = 0.1;
%! S = 1 ./ (s + (1 - s) ./ p);
%! f = omp.bench.amdahl_fit (p, S);
%! assert (f.serial_fraction, s, 1e-12);
%! assert (f.max_speedup, 10, 1e-9);
%! assert (f.rmse < 1e-12);

%!test
%! ## Perfect linear scaling -> zero serial fraction.
%! f = omp.bench.amdahl_fit ([1 2 4], [1 2 4]);
%! assert (f.serial_fraction, 0, 1e-12);
%! assert (isinf (f.max_speedup));

%!test
%! ## Superlinear speedups are clipped to s = 0, not negative.
%! f = omp.bench.amdahl_fit ([1 2 4], [1 2.5 5]);
%! assert (f.serial_fraction, 0);

%!error omp.bench.amdahl_fit (1, 1)
