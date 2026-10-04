% Location: mini_projects/signal_processing_simulation/private/peak_detection_demo.m
% (split out of spectrum_analyzer.m so it can be called from other files)

function peak_detection_demo(signal, fs)
    % Demonstrate peak detection in frequency domain
    
    [freq, mag, ~] = analyze_spectrum(signal, fs, 'plot', false);
    
    % Find peaks
    % Octave's findpeaks rejects negative data, so search the spectrum
    % (in dB) shifted to be non-negative and shift the peaks back
    floor_db = min(mag);
    [peaks, locs] = findpeaks(mag - floor_db, 'MinPeakHeight', max(eps, max(mag)-30 - floor_db), 'MinPeakDistance', 10);
    peaks = peaks + floor_db;
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
