% Location: mini_projects/signal_processing_simulation/spectrum_analyzer.m
% FFT Analysis and Spectral Visualization Tools

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

function plot_spectrum_analysis(signal, frequencies, magnitude, phase, fs, window_type)
    % Plot comprehensive spectrum analysis results
    
    figure('Position', [100, 100, 1400, 1000]);
    
    % Time domain signal
    subplot(3, 2, 1);
    t = (0:length(signal)-1) / fs;
    plot(t, signal, 'b', 'LineWidth', 1);
    title('Time Domain Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Magnitude spectrum (linear)
    subplot(3, 2, 2);
    plot(frequencies, 10.^(magnitude/20), 'r', 'LineWidth', 1);
    title('Magnitude Spectrum (Linear Scale)');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    grid on;
    
    % Magnitude spectrum (dB)
    subplot(3, 2, 3);
    semilogx(frequencies, magnitude, 'g', 'LineWidth', 1.5);
    title('Magnitude Spectrum (dB)');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    
    % Phase spectrum
    subplot(3, 2, 4);
    semilogx(frequencies, unwrap(phase) * 180/pi, 'm', 'LineWidth', 1);
    title('Phase Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Phase (degrees)');
    grid on;
    
    % Power Spectral Density
    subplot(3, 2, 5);
    psd = magnitude - 10*log10(fs); % Convert to PSD
    semilogx(frequencies, psd, 'c', 'LineWidth', 1.5);
    title('Power Spectral Density');
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB/Hz)');
    grid on;
    
    % 3D Waterfall (if signal is long enough)
    subplot(3, 2, 6);
    if length(signal) > 1024
        [S, F, T] = spectrogram(signal, 256, 128, 512, fs);
        surf(T, F, 20*log10(abs(S) + eps), 'EdgeColor', 'none');
        view(45, 45);
        title('Spectrogram (3D View)');
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        zlabel('Magnitude (dB)');
        colorbar;
    else
        % Just show a simple bar plot of peak frequencies
        [peaks, locs] = findpeaks(magnitude, 'MinPeakHeight', max(magnitude)-20, 'NPeaks', 10);
        stem(frequencies(locs), peaks, 'filled');
        title('Peak Frequencies');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (dB)');
        grid on;
    end
    
    sgtitle(sprintf('Spectrum Analysis (Window: %s)', window_type));
end

function compare_windows(signal, fs)
    % Compare different window functions
    
    windows = {'rect', 'hann', 'hamming', 'blackman'};
    colors = {'k', 'b', 'r', 'g'};
    
    figure('Position', [100, 100, 1400, 800]);
    
    % Window functions
    subplot(2, 2, 1);
    N = 256;
    for i = 1:length(windows)
        switch windows{i}
            case 'rect'
                w = ones(N, 1);
            case 'hann'
                w = hann(N);
            case 'hamming'
                w = hamming(N);
            case 'blackman'
                w = blackman(N);
        end
        plot(w, colors{i}, 'LineWidth', 1.5);
        hold on;
    end
    legend(windows, 'Location', 'best');
    title('Window Functions');
    xlabel('Sample');
    ylabel('Amplitude');
    grid on;
    
    % Windowed signals
    subplot(2, 2, 2);
    n_samples = min(500, length(signal));
    t = (0:n_samples-1) / fs;
    for i = 1:length(windows)
        switch windows{i}
            case 'rect'
                w = ones(n_samples, 1);
            case 'hann'
                w = hann(n_samples);
            case 'hamming'
                w = hamming(n_samples);
            case 'blackman'
                w = blackman(n_samples);
        end
        windowed = signal(1:n_samples) .* w;
        plot(t, windowed, colors{i}, 'LineWidth', 1);
        hold on;
    end
    legend(windows, 'Location', 'best');
    title('Windowed Signals');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Spectrum comparison
    subplot(2, 2, [3, 4]);
    for i = 1:length(windows)
        [freq, mag, ~] = analyze_spectrum(signal, fs, 'window', windows{i}, 'plot', false);
        semilogx(freq, mag, colors{i}, 'LineWidth', 1.5);
        hold on;
    end
    legend(windows, 'Location', 'best');
    title('Magnitude Spectra Comparison');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    
    sgtitle('Window Function Comparison');
end

function spectrogram_analysis(signal, fs, varargin)
    % Advanced spectrogram analysis
    
    % Default parameters
    window_length = 256;
    overlap = 128;
    nfft = 512;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'window_length'
                window_length = varargin{i+1};
            case 'overlap'
                overlap = varargin{i+1};
            case 'nfft'
                nfft = varargin{i+1};
        end
    end
    
    [S, F, T] = spectrogram(signal, window_length, overlap, nfft, fs);
    S_db = 20*log10(abs(S) + eps);
    
    figure('Position', [100, 100, 1400, 800]);
    
    % 2D Spectrogram
    subplot(2, 2, 1);
    imagesc(T, F, S_db);
    axis xy;
    colorbar;
    title('Spectrogram (2D)');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    
    % 3D Surface
    subplot(2, 2, 2);
    surf(T, F, S_db, 'EdgeColor', 'none');
    title('Spectrogram (3D)');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    zlabel('Magnitude (dB)');
    view(45, 45);
    
    % Time-averaged spectrum
    subplot(2, 2, 3);
    avg_spectrum = mean(S_db, 2);
    semilogx(F, avg_spectrum, 'b', 'LineWidth', 2);
    title('Time-Averaged Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Average Magnitude (dB)');
    grid on;
    
    % Frequency-averaged power vs time
    subplot(2, 2, 4);
    power_vs_time = mean(10.^(S_db/10), 1);
    plot(T, 10*log10(power_vs_time), 'r', 'LineWidth', 2);
    title('Total Power vs Time');
    xlabel('Time (s)');
    ylabel('Power (dB)');
    grid on;
    
    sgtitle('Advanced Spectrogram Analysis');
end

function peak_detection_demo(signal, fs)
    % Demonstrate peak detection in frequency domain
    
    [freq, mag, ~] = analyze_spectrum(signal, fs, 'plot', false);
    
    % Find peaks
    [peaks, locs] = findpeaks(mag, 'MinPeakHeight', max(mag)-30, 'MinPeakDistance', 10);
    peak_freqs = freq(locs);
    
    figure('Position', [100, 100, 1200, 600]);
    
    subplot(1, 2, 1);
    plot(freq, mag, 'b', 'LineWidth', 1);
    hold on;
    plot(peak_freqs, peaks, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    title('Peak Detection in Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    legend('Spectrum', 'Peaks', 'Location', 'best');
    
    % Add text annotations for peak frequencies
    for i = 1:length(peak_freqs)
        text(peak_freqs(i), peaks(i)+2, sprintf('%.1f Hz', peak_freqs(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    
    subplot(1, 2, 2);
    bar(peak_freqs, 10.^(peaks/20));
    title('Peak Amplitudes');
    xlabel('Frequency (Hz)');
    ylabel('Linear Amplitude');
    grid on;
    
    % Display peak information
    fprintf('\nDetected Peaks:\n');
    fprintf('Frequency (Hz)\tMagnitude (dB)\tLinear Amplitude\n');
    fprintf('----------------------------------------------\n');
    for i = 1:length(peak_freqs)
        fprintf('%10.2f\t%12.2f\t%15.4f\n', peak_freqs(i), peaks(i), 10^(peaks(i)/20));
    end
end

function demo_spectrum_analyzer()
    % Comprehensive demonstration of spectrum analyzer
    
    fprintf('Spectrum Analyzer Demonstration\n');
    fprintf('===============================\n\n');
    
    % Generate test signals
    fs = 1000;
    t = 0:1/fs:5;
    
    % Multi-component signal
    signal1 = sin(2*pi*50*t) + 0.7*sin(2*pi*120*t) + 0.4*sin(2*pi*200*t) + 0.1*randn(size(t));
    
    % Chirp signal
    signal2 = chirp(t, 20, 5, 200) + 0.05*randn(size(t));
    
    % Analyze multi-component signal
    fprintf('1. Analyzing multi-component signal (50, 120, 200 Hz)...\n');
    [freq1, mag1, phase1] = analyze_spectrum(signal1, fs);
    
    fprintf('2. Comparing window functions...\n');
    compare_windows(signal1, fs);
    
    fprintf('3. Peak detection demonstration...\n');
    peak_detection_demo(signal1, fs);
    
    fprintf('4. Analyzing chirp signal...\n');
    analyze_spectrum(signal2, fs, 'window', 'blackman');
    
    fprintf('5. Spectrogram analysis of chirp...\n');
    spectrogram_analysis(signal2, fs, 'window_length', 128, 'overlap', 64);
    
    fprintf('\nSpectrum Analyzer demonstration complete!\n');
end