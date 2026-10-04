% Location: mini_projects/signal_processing_simulation/private/compare_windows.m
% (split out of spectrum_analyzer.m so it can be called from other files)

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
