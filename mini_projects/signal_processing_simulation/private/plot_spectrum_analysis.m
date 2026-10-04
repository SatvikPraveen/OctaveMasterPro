% Location: mini_projects/signal_processing_simulation/private/plot_spectrum_analysis.m
% (split out of spectrum_analyzer.m so it can be called from other files)

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
        % Octave's findpeaks rejects negative data: shift the dB spectrum
        floor_db = min(magnitude);
        [peaks, locs] = findpeaks(magnitude - floor_db, 'MinPeakHeight', max(eps, max(magnitude)-20 - floor_db));
        [~, order] = sort(peaks, 'descend');
        keep = sort(order(1:min(10, numel(order))));
        peaks = peaks(keep) + floor_db;
        locs = locs(keep);
        stem(frequencies(locs), peaks, 'filled');
        title('Peak Frequencies');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (dB)');
        grid on;
    end
    
    sgtitle(sprintf('Spectrum Analysis (Window: %s)', window_type));
end
