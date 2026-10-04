% Location: mini_projects/signal_processing_simulation/private/spectrogram.m
% Short-time Fourier transform with MATLAB's spectrogram() call signature.
%
% The Octave signal package provides specgram() but not spectrogram(),
% so the spectrum analyzer uses this local implementation.
%
%   [S, F, T] = spectrogram(x, window, noverlap, nfft, fs)
%
%   window    window vector, or a length (a Hann window is used)
%   noverlap  overlapping samples between segments (default: half window)
%   nfft      FFT length (default: max(256, next power of 2 >= window))
%   fs        sampling frequency (default 1)
%
% S has one column per segment (one-sided for real x), F holds the
% frequencies in Hz and T the segment centre times in seconds.  Without
% output arguments the spectrogram is plotted in dB.

function [S, F, T] = spectrogram(x, win, noverlap, nfft, fs)
    x = x(:);
    if nargin < 2 || isempty(win)
        win = fix(numel(x) / 4.5);
    end
    if isscalar(win)
        win = hanning(win);
    end
    win = win(:);
    L = numel(win);
    if nargin < 3 || isempty(noverlap)
        noverlap = fix(L / 2);
    end
    if nargin < 4 || isempty(nfft)
        nfft = max(256, 2^nextpow2(L));
    end
    if nargin < 5 || isempty(fs)
        fs = 1;
    end
    if numel(x) < L
        error('spectrogram: signal is shorter than the window');
    end

    hop = L - noverlap;
    ncol = fix((numel(x) - noverlap) / hop);
    idx = (1:L)' + (0:ncol-1) * hop;
    frames = x(idx) .* win;
    X = fft(frames, nfft);
    if isreal(x)
        nf = fix(nfft / 2) + 1;
    else
        nf = nfft;
    end
    X = X(1:nf, :);
    f = (0:nf-1)' * fs / nfft;
    t = ((0:ncol-1) * hop + L / 2) / fs;

    if nargout == 0
        imagesc(t, f, 20 * log10(abs(X) + eps));
        axis xy;
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        colorbar;
    else
        S = X;
        F = f;
        T = t;
    end
end
