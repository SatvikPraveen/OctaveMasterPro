% Location: mini_projects/signal_processing_simulation/private/spectrogram_analysis.m
% (split out of spectrum_analyzer.m so it can be called from other files)

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
