% Location: mini_projects/signal_processing_simulation/private/demo_filtering.m
% (split out of filter_design.m so it can be called from other files)

function demo_filtering()
    % Demonstrate filter design and application
    
    % Generate test signal with multiple frequencies
    fs = 1000;
    t = 0:1/fs:2;
    
    % Composite signal: 10Hz + 50Hz + 150Hz + noise
    signal = sin(2*pi*10*t) + 0.5*sin(2*pi*50*t) + 0.3*sin(2*pi*150*t) + 0.1*randn(size(t));
    
    % Apply different filter types
    [low_passed, lp_coeffs] = apply_filter(signal, 'lowpass', 75, fs, 6, 'butter');
    [high_passed, hp_coeffs] = apply_filter(signal, 'highpass', 25, fs, 6, 'butter');
    [band_passed, bp_coeffs] = apply_filter(signal, 'bandpass', [30, 80], fs, 4, 'butter');
    [band_stopped, bs_coeffs] = apply_filter(signal, 'bandstop', [40, 60], fs, 4, 'butter');
    
    % Plot results
    figure('Position', [100, 100, 1200, 900]);
    
    % Time domain
    subplot(3, 2, 1);
    plot(t(1:500), signal(1:500), 'k', 'LineWidth', 1.5);
    title('Original Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    subplot(3, 2, 2);
    plot(t(1:500), low_passed(1:500), 'b', 'LineWidth', 1.5);
    title('Low-pass Filtered (< 75 Hz)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    subplot(3, 2, 3);
    plot(t(1:500), high_passed(1:500), 'r', 'LineWidth', 1.5);
    title('High-pass Filtered (> 25 Hz)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    subplot(3, 2, 4);
    plot(t(1:500), band_passed(1:500), 'g', 'LineWidth', 1.5);
    title('Band-pass Filtered (30-80 Hz)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    subplot(3, 2, 5);
    plot(t(1:500), band_stopped(1:500), 'm', 'LineWidth', 1.5);
    title('Band-stop Filtered (40-60 Hz blocked)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Frequency domain comparison
    subplot(3, 2, 6);
    N = length(signal);
    f = (0:N-1) * fs / N;
    f = f(1:N/2);
    
    X_orig = abs(fft(signal));
    X_lp = abs(fft(low_passed));
    X_hp = abs(fft(high_passed));
    X_bp = abs(fft(band_passed));
    X_bs = abs(fft(band_stopped));
    
    semilogx(f, 20*log10(X_orig(1:N/2)), 'k', 'LineWidth', 1.5);
    hold on;
    semilogx(f, 20*log10(X_lp(1:N/2)), 'b', 'LineWidth', 1);
    semilogx(f, 20*log10(X_hp(1:N/2)), 'r', 'LineWidth', 1);
    semilogx(f, 20*log10(X_bp(1:N/2)), 'g', 'LineWidth', 1);
    semilogx(f, 20*log10(X_bs(1:N/2)), 'm', 'LineWidth', 1);
    
    legend('Original', 'Low-pass', 'High-pass', 'Band-pass', 'Band-stop', 'Location', 'best');
    title('Frequency Domain Comparison');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    xlim([1, fs/2]);
    
    sgtitle('Digital Filter Demonstration');
end
