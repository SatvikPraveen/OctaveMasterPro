% Location: mini_projects/signal_processing_simulation/private/compare_filter_methods.m
% (split out of filter_design.m so it can be called from other files)

function compare_filter_methods(signal, cutoff_freq, fs)
    % Compare different filter design methods
    
    methods = {'butter', 'cheby1', 'cheby2', 'ellip'};
    colors = {'b', 'r', 'g', 'm'};
    
    figure('Position', [100, 100, 1400, 800]);
    
    % Time domain comparison
    subplot(2, 2, 1);
    t = (0:length(signal)-1) / fs;
    plot(t, signal, 'k', 'LineWidth', 1.5);
    hold on;
    
    for i = 1:length(methods)
        if strcmp(methods{i}, 'cheby1')
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i}, 1);
        elseif strcmp(methods{i}, 'cheby2')
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i}, 40);
        elseif strcmp(methods{i}, 'ellip')
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i}, 1, 40);
        else
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i});
        end
        plot(t, filtered, colors{i}, 'LineWidth', 1);
    end
    
    legend(['Original', methods], 'Location', 'best');
    title('Time Domain Comparison');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Frequency responses
    subplot(2, 2, 2);
    for i = 1:length(methods)
        if strcmp(methods{i}, 'cheby1')
            [b, a] = cheby1(4, 1, cutoff_freq/(fs/2), 'low');
        elseif strcmp(methods{i}, 'cheby2')
            [b, a] = cheby2(4, 40, cutoff_freq/(fs/2), 'low');
        elseif strcmp(methods{i}, 'ellip')
            [b, a] = ellip(4, 1, 40, cutoff_freq/(fs/2), 'low');
        else
            [b, a] = butter(4, cutoff_freq/(fs/2), 'low');
        end
        
        [h, w] = freqz(b, a, 1024, fs);
        semilogx(w, 20*log10(abs(h)), colors{i}, 'LineWidth', 1.5);
        hold on;
    end
    
    legend(methods, 'Location', 'best');
    title('Magnitude Response Comparison');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    ylim([-80, 5]);
    
    % Original signal spectrum
    subplot(2, 2, 3);
    N = length(signal);
    f = (0:N-1) * fs / N;
    f = f(1:N/2);
    X = fft(signal);
    X_mag = abs(X(1:N/2));
    
    semilogx(f, 20*log10(X_mag), 'k', 'LineWidth', 1.5);
    title('Original Signal Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    
    % Filtered signal spectra
    subplot(2, 2, 4);
    for i = 1:length(methods)
        if strcmp(methods{i}, 'cheby1')
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i}, 1);
        elseif strcmp(methods{i}, 'cheby2')
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i}, 40);
        elseif strcmp(methods{i}, 'ellip')
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i}, 1, 40);
        else
            filtered = apply_filter(signal, 'lowpass', cutoff_freq, fs, 4, methods{i});
        end
        
        Y = fft(filtered);
        Y_mag = abs(Y(1:N/2));
        semilogx(f, 20*log10(Y_mag), colors{i}, 'LineWidth', 1.5);
        hold on;
    end
    
    legend(methods, 'Location', 'best');
    title('Filtered Signal Spectra');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    
    sgtitle(sprintf('Filter Method Comparison (Cutoff: %d Hz)', cutoff_freq));
end
