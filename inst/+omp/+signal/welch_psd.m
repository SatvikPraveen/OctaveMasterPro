function [P, f, info] = welch_psd (x, fs, varargin)
  ## -*- texinfo -*-
  ## @deftypefn {} {[@var{P}, @var{f}] =} omp.signal.welch_psd (@var{x}, @var{fs})
  ## @deftypefnx {} {[@var{P}, @var{f}, @var{info}] =} omp.signal.welch_psd (@var{x}, @var{fs}, "SegmentLength", @var{L}, "Overlap", @var{r}, "Window", @var{w}, "Detrend", @var{d})
  ## One-sided power spectral density of the real signal @var{x} sampled at
  ## @var{fs} Hz by Welch's method of averaged modified periodograms
  ## (Welch, 1967).
  ##
  ## The signal is split into segments of @var{L} samples (default: the
  ## largest power of two giving at least 8 segments, minimum 16) overlapping
  ## by the fraction @var{r} (default 0.5).  Each segment is detrended
  ## (@qcode{"constant"} (default), @qcode{"linear"} or @qcode{"none"}),
  ## multiplied by the window @var{w} (@qcode{"hann"} (default, periodic),
  ## @qcode{"hamming"}, @qcode{"rect"}, or a numeric vector of length L),
  ## and its periodogram computed by FFT.
  ##
  ## Scaling is a density, in units^2/Hz:
  ## P(f) = 2 |X(f)|^2 / (fs sum(w^2)) for 0 < f < fs/2 (DC and Nyquist not
  ## doubled), so that @code{sum (P) * (f(2) - f(1))} approximates the
  ## variance of the detrended signal (Parseval).
  ##
  ## @var{info} reports the number of segments K, the frequency resolution
  ## df = fs/L, the equivalent noise bandwidth ENBW = fs sum(w^2)/sum(w)^2
  ## and the approximate equivalent degrees of freedom of each PSD
  ## estimate, nu ~ 2K for non-overlapping segments (fewer with overlap),
  ## which governs the chi-square confidence interval
  ## nu P / chi2_(nu,1-a/2) <= S(f) <= nu P / chi2_(nu,a/2).
  ## @end deftypefn

  x = x(:);
  N = numel (x);
  opts = struct ("segmentlength", [], "overlap", 0.5, "window", "hann", ...
                 "detrend", "constant");
  for k = 1:2:numel (varargin)
    key = lower (varargin{k});
    if (! isfield (opts, key))
      error ("omp.signal.welch_psd: unknown option '%s'", varargin{k});
    endif
    opts.(key) = varargin{k+1};
  endfor

  L = opts.segmentlength;
  if (isempty (L))
    L = max (16, 2 ^ floor (log2 (N / 4.5)));   # >= 8 segments at 50% overlap
    L = min (L, N);
  endif
  if (L > N || L < 2)
    error ("omp.signal.welch_psd: SegmentLength must be in [2, numel(x)]");
  endif
  if (opts.overlap < 0 || opts.overlap >= 1)
    error ("omp.signal.welch_psd: Overlap must be in [0, 1)");
  endif

  if (isnumeric (opts.window))
    w = opts.window(:);
    if (numel (w) != L)
      error ("omp.signal.welch_psd: window length must equal SegmentLength");
    endif
  else
    n = (0:L-1)';
    switch (lower (opts.window))
      case "hann"
        w = 0.5 - 0.5 * cos (2 * pi * n / L);
      case "hamming"
        w = 0.54 - 0.46 * cos (2 * pi * n / L);
      case {"rect", "boxcar", "none"}
        w = ones (L, 1);
      otherwise
        error ("omp.signal.welch_psd: unknown window '%s'", opts.window);
    endswitch
  endif

  step = max (1, round (L * (1 - opts.overlap)));
  starts = 1:step:(N - L + 1);
  K = numel (starts);
  idx = starts + (0:L-1)';              # L-by-K
  S = x(idx);
  switch (lower (opts.detrend))
    case "constant"
      S -= mean (S, 1);
    case "linear"
      t = [ones(L, 1), (0:L-1)' / L];
      S -= t * (t \ S);
    case "none"
    otherwise
      error ("omp.signal.welch_psd: unknown detrend '%s'", opts.detrend);
  endswitch

  X = fft (S .* w);
  nf = floor (L / 2) + 1;
  Pk = abs (X(1:nf, :)) .^ 2 / (fs * sum (w .^ 2));
  P = mean (Pk, 2);
  if (mod (L, 2) == 0)
    P(2:end-1) *= 2;                    # Nyquist bin is unique for even L
  else
    P(2:end) *= 2;
  endif
  f = (0:nf-1)' * fs / L;

  ## Welch (1967) variance-reduction factor for overlapping segments.
  rho = 0;
  ov = L - step;
  if (ov > 0)
    rho = (sum (w(1:ov) .* w(end-ov+1:end)) / sum (w .^ 2)) ^ 2;
  endif
  nu = 2 * K / (1 + 2 * rho * (K - 1) / K);
  info = struct ("segments", K, "segment_length", L, "df", fs / L, ...
                 "enbw", fs * sum (w .^ 2) / sum (w) ^ 2, "dof", nu);
endfunction

%!test
%! ## Parseval: integrated PSD equals the variance of white noise, and the
%! ## one-sided level is 2 sigma^2 / fs.
%! omp.repro.seed (1);
%! fs = 1000; sigma = 2;
%! x = sigma * randn (2 ^ 16, 1);
%! [P, f] = omp.signal.welch_psd (x, fs, "SegmentLength", 1024);
%! df = f(2) - f(1);
%! assert (sum (P) * df, var (x), 0.03 * var (x));
%! assert (median (P(2:end-1)), 2 * sigma ^ 2 / fs, 0.05 * 2 * sigma ^ 2 / fs);

%!test
%! ## A sinusoid of amplitude A carries power A^2/2 at its frequency.
%! fs = 512; t = (0:fs*16-1)' / fs; A = 3; f0 = 50;
%! x = A * sin (2 * pi * f0 * t);
%! for win = {"hann", "hamming", "rect"}
%!   [P, f] = omp.signal.welch_psd (x, fs, "SegmentLength", 512, "Window", win{1});
%!   [~, k] = max (P);
%!   assert (f(k), f0);
%!   band = abs (f - f0) <= 4;
%!   assert (sum (P(band)) * (f(2) - f(1)), A ^ 2 / 2, 0.01 * A ^ 2 / 2);
%! endfor

%!test
%! ## Odd segment length and the degrees-of-freedom bookkeeping.
%! omp.repro.seed (2);
%! x = randn (5000, 1);
%! [P, f, info] = omp.signal.welch_psd (x, 100, "SegmentLength", 201, "Overlap", 0);
%! assert (numel (f), 101);
%! assert (info.segments, 24);
%! assert (info.dof, 48);
%! assert (sum (P) * info.df, var (x), 0.1);

%!test
%! ## Colored noise: AR(1) spectrum matches the analytic shape.
%! omp.repro.seed (3);
%! phi = 0.8; fs = 1;
%! x = filter (1, [1 -phi], randn (2 ^ 17, 1));
%! [P, f] = omp.signal.welch_psd (x, fs, "SegmentLength", 256);
%! S = 2 ./ abs (1 - phi * exp (-2i * pi * f)) .^ 2;    # one-sided, sigma^2 = 1
%! k = 2:numel (f) - 1;
%! assert (median (P(k) ./ S(k)), 1, 0.05);

%!testif HAVE_FFTW
%! ## Independent reference: Octave-Forge signal package pwelch.
%! if (isempty (pkg ("list", "signal")))
%!   return;
%! endif
%! pkg load signal
%! omp.repro.seed (5);
%! x = randn (4096, 1);
%! [P, f] = omp.signal.welch_psd (x, 100, "SegmentLength", 256);
%! w = 0.5 - 0.5 * cos (2 * pi * (0:255)' / 256);
%! [P2, f2] = pwelch (x, w, 0.5, 256, 100, "onesided", "mean");
%! assert (P, P2, 1e-12 * max (P2));
%! assert (f, f2);

%!error <SegmentLength> omp.signal.welch_psd (randn (10, 1), 1, "SegmentLength", 20)
%!error <unknown window> omp.signal.welch_psd (randn (64, 1), 1, "Window", "foo")
