% Location: mini_projects/signal_processing_simulation/private/analyze_spectrum.m
% (split out of spectrum_analyzer.m so it can be called from other files)

function [frequencies, magnitude, phase] = analyze_spectrum(signal, fs, varargin)
    % Perform comprehensive spectral analysis of input signal
    %
    % Inputs:
    %   signal - input signal vector
    %   fs - sampling frequency (Hz)
    %   varargin - optional parameters
    %     window_type - 'hann', 'hamming', 'blackman', 'rect' (default: 'hann')
    %     nfft - FFT length (default: next power of 2)
    %     overlap - window overlap percentage (default: 50)
    %     plot_flag - boolean to plot results (default: true)
    %
    % Outputs:
    %   frequencies - frequency vector (Hz)
    %   magnitude - magnitude spectrum (dB)
    %   phase - phase spectrum (radians)
    
    % Default parameters
    window_type = 'hann';
    nfft = 2^nextpow2(length(signal));
    overlap_percent = 50;
    plot_flag = true;
    detrend_flag = true;
    
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'window'
                window_type = varargin{i+1};
            case 'nfft'
                nfft = varargin{i+1};
            case 'overlap'
                overlap_percent = varargin{i+1};
            case 'plot'
                plot_flag = varargin{i+1};
            case 'detrend'
                detrend_flag = varargin{i+1};
        end
    end
    
    % Detrend signal if requested
    if detrend_flag
        signal = detrend(signal);
    end
    
    % Apply windowing
    N = length(signal);
    switch lower(window_type)
        case 'hann'
            window = hann(N);
        case 'hamming'
            window = hamming(N);
        case 'blackman'
            window = blackman(N);
        case 'rect'
            window = ones(N, 1);
        otherwise
            warning('Unknown window type, using Hann');
            window = hann(N);
    end
    
    % Apply window and compute FFT
    windowed_signal = signal(:) .* window(:);
    X = fft(windowed_signal, nfft);
    
    % Compute frequency vector
    frequencies = (0:nfft-1) * fs / nfft;
    frequencies = frequencies(1:floor(nfft/2)+1);
    
    % Compute magnitude and phase
    X_half = X(1:length(frequencies));
    magnitude = 20 * log10(abs(X_half) + eps); % Add small value to avoid log(0)
    phase = angle(X_half);
    
    % Normalize magnitude for window effect
    window_gain = sum(window) / N;
    magnitude = magnitude - 20*log10(window_gain);
    
    if plot_flag
        plot_spectrum_analysis(signal, frequencies, magnitude, phase, fs, window_type);
    end
end
