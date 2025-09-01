% Location: mini_projects/signal_processing_simulation/signal_demo.m
% Main Signal Processing Demonstration Script

function signal_demo()
    % Complete signal processing simulation demonstration
    
    clear; clc; close all;
    
    fprintf('====================================================\n');
    fprintf('    SIGNAL PROCESSING SIMULATION DEMONSTRATION     \n');
    fprintf('====================================================\n\n');
    
    % Set random seed for reproducible results
    rand('state', 42);
    randn('state', 42);
    
    try
        % Main demonstration menu
        while true
            fprintf('\nSelect a demonstration:\n');
            fprintf('1. Signal Generation Demo\n');
            fprintf('2. Filter Design Demo\n');
            fprintf('3. Spectrum Analysis Demo\n');
            fprintf('4. Complete Pipeline Demo\n');
            fprintf('5. Interactive Signal Lab\n');
            fprintf('0. Exit\n');
            
            choice = input('Enter your choice (0-5): ');
            
            switch choice
                case 0
                    fprintf('\nExiting Signal Processing Demo. Goodbye!\n');
                    break;
                    
                case 1
                    signal_generation_demo();
                    
                case 2
                    filter_design_demo();
                    
                case 3
                    spectrum_analysis_demo();
                    
                case 4
                    complete_pipeline_demo();
                    
                case 5
                    interactive_signal_lab();
                    
                otherwise
                    fprintf('Invalid choice. Please select 0-5.\n');
            end
            
            if choice ~= 0
                input('\nPress Enter to continue...');
            end
        end
        
    catch err
        fprintf('Error in signal_demo: %s\n', err.message);
        fprintf('Make sure all required functions are in the current directory.\n');
    end
end

function signal_generation_demo()
    % Demonstrate signal generation capabilities
    
    fprintf('\n--- Signal Generation Demonstration ---\n');
    
    % Parameters
    fs = 1000; % Hz
    duration = 2; % seconds
    frequency = 50; % Hz
    amplitude = 1;
    
    % Generate various signals
    fprintf('Generating different signal types...\n');
    
    [t, sine_wave] = generate_signal('sine', fs, duration, frequency, amplitude);
    [~, square_wave] = generate_signal('square', fs, duration, frequency, amplitude, 30);
    [~, sawtooth_wave] = generate_signal('sawtooth', fs, duration, frequency, amplitude);
    [~, chirp_signal] = generate_signal('chirp', fs, duration, 10, amplitude, 100);
    [~, noise_signal] = generate_signal('noise', fs, duration, frequency, amplitude, 'white');
    
    % Composite signal with harmonics
    freqs = [20, 50, 100, 150];
    amps = [1, 0.7, 0.4, 0.2];
    [~, composite_signal] = generate_signal('composite', fs, duration, 20, 1, freqs, amps);
    
    % Plot all signals
    figure('Position', [50, 50, 1400, 900]);
    
    signals = {sine_wave, square_wave, sawtooth_wave, chirp_signal, noise_signal, composite_signal};
    titles = {'Sine Wave (50 Hz)', 'Square Wave (30% duty)', 'Sawtooth Wave', ...
              'Chirp (10-100 Hz)', 'White Noise', 'Composite Signal'};
    
    for i = 1:6
        subplot(3, 2, i);
        if i <= 3 || i == 6
            plot(t(1:500), signals{i}(1:500), 'LineWidth', 1.5);
        else
            plot(t, signals{i}, 'LineWidth', 1.5);
        end
        title(titles{i});
        xlabel('Time (s)');
        ylabel('Amplitude');
        grid on;
    end
    
    sgtitle('Signal Generation Demonstration');
    
    fprintf('Signal generation complete. Check the plot.\n');
end

function filter_design_demo()
    % Demonstrate digital filter design and application
    
    fprintf('\n--- Filter Design Demonstration ---\n');
    
    % Create a noisy signal with multiple frequency components
    fs = 1000;
    t = 0:1/fs:3;
    
    % Clean signal: 20Hz + 60Hz + 150Hz
    clean_signal = sin(2*pi*20*t) + 0.8*sin(2*pi*60*t) + 0.5*sin(2*pi*150*t);
    
    % Add high-frequency noise
    noise = 0.3 * sin(2*pi*300*t) + 0.2 * sin(2*pi*400*t) + 0.1*randn(size(t));
    noisy_signal = clean_signal + noise;
    
    fprintf('Created test signal with components at 20, 60, 150 Hz plus noise...\n');
    
    % Design and apply various filters
    fprintf('Applying different filter types...\n');
    
    % Low-pass filter (remove high-frequency noise)
    [lp_filtered, lp_coeffs] = apply_filter(noisy_signal, 'lowpass', 200, fs, 6, 'butter');
    
    % Band-pass filter (extract 60Hz component)
    [bp_filtered, bp_coeffs] = apply_filter(noisy_signal, 'bandpass', [50, 70], fs, 4, 'butter');
    
    % Band-stop filter (remove 60Hz component - power line interference)
    [bs_filtered, bs_coeffs] = apply_filter(noisy_signal, 'bandstop', [58, 62], fs, 4, 'butter');
    
    % High-pass filter (remove DC and low frequencies)
    [hp_filtered, hp_coeffs] = apply_filter(noisy_signal, 'highpass', 40, fs, 4, 'butter');
    
    % Plot results
    figure('Position', [100, 100, 1400, 1000]);
    
    % Time domain plots
    subplot(3, 2, 1);
    plot(t(1:1000), clean_signal(1:1000), 'g', 'LineWidth', 2);
    hold on;
    plot(t(1:1000), noisy_signal(1:1000), 'r', 'LineWidth', 1);
    legend('Clean Signal', 'Noisy Signal', 'Location', 'best');
    title('Original Signals');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    subplot(3, 2, 2);
    plot(t(1:1000), lp_filtered(1:1000), 'b', 'LineWidth', 1.5);
    title('Low-pass Filtered (< 200 Hz)');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    subplot(3, 2, 3);
    plot(t(1:1000), bp_filtered(1:1000), 'm', 'LineWidth', 1.5);
    title('Band-pass Filtered (50-70 Hz)');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    subplot(3, 2, 4);
    plot(t(1:1000), bs_filtered(1:1000), 'c', 'LineWidth', 1.5);
    title('Band-stop Filtered (58-62 Hz removed)');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    subplot(3, 2, 5);
    plot(t(1:1000), hp_filtered(1:1000), 'k', 'LineWidth', 1.5);
    title('High-pass Filtered (> 40 Hz)');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    % Frequency domain comparison
    subplot(3, 2, 6);
    N = length(noisy_signal);
    f = (0:N-1) * fs / N;
    f = f(1:N/2);
    
    X_noisy = abs(fft(noisy_signal));
    X_lp = abs(fft(lp_filtered));
    X_bp = abs(fft(bp_filtered));
    X_bs = abs(fft(bs_filtered));
    
    semilogx(f, 20*log10(X_noisy(1:N/2)), 'r', 'LineWidth', 1.5); hold on;
    semilogx(f, 20*log10(X_lp(1:N/2)), 'b', 'LineWidth', 1.5);
    semilogx(f, 20*log10(X_bp(1:N/2)), 'm', 'LineWidth', 1.5);
    semilogx(f, 20*log10(X_bs(1:N/2)), 'c', 'LineWidth', 1.5);
    
    legend('Noisy', 'Low-pass', 'Band-pass', 'Band-stop', 'Location', 'best');
    title('Frequency Domain Comparison');
    xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)'); grid on;
    xlim([1, fs/2]);
    
    sgtitle('Digital Filter Design Demonstration');
    
    fprintf('Filter design demonstration complete.\n');
end

function spectrum_analysis_demo()
    % Demonstrate spectrum analysis capabilities
    
    fprintf('\n--- Spectrum Analysis Demonstration ---\n');
    
    % Call the spectrum analyzer demo
    demo_spectrum_analyzer();
end

function complete_pipeline_demo()
    % Demonstrate complete signal processing pipeline
    
    fprintf('\n--- Complete Signal Processing Pipeline ---\n');
    
    % Simulate a real-world scenario: ECG signal processing
    fprintf('Simulating ECG signal processing pipeline...\n');
    
    fs = 500; % Hz (typical for ECG)
    t = 0:1/fs:10; % 10 seconds of data
    
    % Generate synthetic ECG-like signal
    fprintf('1. Generating synthetic ECG signal...\n');
    heart_rate = 72; % BPM
    ecg_freq = heart_rate / 60; % Hz
    
    % Basic ECG waveform (simplified)
    ecg_clean = zeros(size(t));
    for i = 1:length(t)
        phase = 2*pi*ecg_freq*t(i);
        % P wave
        ecg_clean(i) = ecg_clean(i) + 0.1*exp(-((mod(phase, 2*pi) - 0.5)^2) / 0.02);
        % QRS complex  
        ecg_clean(i) = ecg_clean(i) + 0.8*exp(-((mod(phase, 2*pi) - pi)^2) / 0.005);
        % T wave
        ecg_clean(i) = ecg_clean(i) + 0.3*exp(-((mod(phase, 2*pi) - 1.8*pi)^2) / 0.08);
    end
    
    % Add realistic noise and artifacts
    fprintf('2. Adding noise and artifacts...\n');
    powerline_noise = 0.1 * sin(2*pi*60*t); % 60Hz power line
    baseline_drift = 0.2 * sin(2*pi*0.1*t); % Breathing artifact
    muscle_noise = 0.05 * randn(size(t)); % Random muscle activity
    motion_artifact = 0.3 * sin(2*pi*0.3*t) .* (t > 3 & t < 4); % Motion during 3-4s
    
    ecg_noisy = ecg_clean + powerline_noise + baseline_drift + muscle_noise + motion_artifact;
    
    % Processing pipeline
    fprintf('3. Applying processing pipeline...\n');
    
    % Step 1: High-pass filter to remove baseline drift
    fprintf('   - Removing baseline drift (high-pass > 0.5 Hz)...\n');
    ecg_step1 = apply_filter(ecg_noisy, 'highpass', 0.5, fs, 2, 'butter');
    
    % Step 2: Low-pass filter to remove high-frequency noise
    fprintf('   - Removing high-frequency noise (low-pass < 50 Hz)...\n');
    ecg_step2 = apply_filter(ecg_step1, 'lowpass', 50, fs, 4, 'butter');
    
    % Step 3: Notch filter to remove 60Hz power line interference
    fprintf('   - Removing 60Hz power line interference...\n');
    ecg_final = apply_filter(ecg_step2, 'bandstop', [58, 62], fs, 4, 'butter');
    
    % Results visualization
    fprintf('4. Visualizing results...\n');
    
    figure('Position', [50, 50, 1400, 800]);
    
    subplot(2, 3, 1);
    plot(t, ecg_clean, 'g', 'LineWidth', 2);
    title('Clean ECG');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    xlim([2, 4]);
    
    subplot(2, 3, 2);
    plot(t, ecg_noisy, 'r', 'LineWidth', 1);
    title('Noisy ECG');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    xlim([2, 4]);
    
    subplot(2, 3, 3);
    plot(t, ecg_final, 'b', 'LineWidth', 1.5);
    title('Processed ECG');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    xlim([2, 4]);
    
    subplot(2, 3, 4);
    plot(t, ecg_clean, 'g', 'LineWidth', 2); hold on;
    plot(t, ecg_final, 'k', 'LineWidth', 1.5);
    legend('Original Clean', 'Processed', 'Location', 'best');
    title('Final Result vs Original');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    xlim([2, 4]);
    
    % Frequency domain comparison
    subplot(2, 3, [5,6]);
    N = length(ecg_noisy);
    f = (0:N-1) * fs / N;
    f = f(1:N/2);
    
    X_noisy = abs(fft(ecg_noisy));
    X_final = abs(fft(ecg_final));
    
    semilogx(f, 20*log10(X_noisy(1:N/2)), 'r', 'LineWidth', 1.5); hold on;
    semilogx(f, 20*log10(X_final(1:N/2)), 'k', 'LineWidth', 1.5);
    legend('Noisy ECG', 'Processed ECG', 'Location', 'best');
    title('Frequency Domain: Before vs After');
    xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)'); grid on;
    xlim([0.1, 100]);
    
    sgtitle('Complete ECG Signal Processing Pipeline');
    
    % Calculate improvement metrics
    fprintf('5. Performance metrics:\n');
    snr_before = 10*log10(var(ecg_clean) / var(ecg_noisy - ecg_clean));
    snr_after = 10*log10(var(ecg_clean) / var(ecg_final - ecg_clean));
    improvement = snr_after - snr_before;
    
    fprintf('   SNR before processing: %.2f dB\n', snr_before);
    fprintf('   SNR after processing:  %.2f dB\n', snr_after);
    fprintf('   SNR improvement:       %.2f dB\n', improvement);
    
    fprintf('Complete pipeline demonstration finished.\n');
end

function filter_design_demo()
    fprintf('\n--- Filter Design Demonstration ---\n');
    demo_filtering();
end

function spectrum_analysis_demo()
    fprintf('\n--- Spectrum Analysis Demonstration ---\n');
    demo_spectrum_analyzer();
end

function interactive_signal_lab()
    % Interactive signal processing laboratory
    
    fprintf('\n--- Interactive Signal Processing Lab ---\n');
    
    while true
        fprintf('\nInteractive Lab Menu:\n');
        fprintf('1. Custom signal generation\n');
        fprintf('2. Filter comparison\n');
        fprintf('3. Window function effects\n');
        fprintf('0. Return to main menu\n');
        
        choice = input('Enter choice: ');
        
        switch choice
            case 0
                break;
            case 1
                custom_signal_generation();
            case 2
                compare_filter_methods_demo();
            case 3
                window_effects_demo();
            otherwise
                fprintf('Invalid choice.\n');
        end
    end
end

function custom_signal_generation()
    fprintf('\n--- Custom Signal Generation ---\n');
    
    fs = input('Enter sampling frequency (Hz) [1000]: ');
    if isempty(fs), fs = 1000; end
    
    duration = input('Enter duration (seconds) [2]: ');
    if isempty(duration), duration = 2; end
    
    fprintf('Signal types: sine, square, sawtooth, chirp, noise\n');
    signal_type = input('Enter signal type [sine]: ', 's');
    if isempty(signal_type), signal_type = 'sine'; end
    
    frequency = input('Enter frequency (Hz) [50]: ');
    if isempty(frequency), frequency = 50; end
    
    [t, signal] = generate_signal(signal_type, fs, duration, frequency, 1);
    
    figure;
    subplot(2, 1, 1);
    plot(t, signal, 'b', 'LineWidth', 1.5);
    title(sprintf('Generated %s Signal', signal_type));
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    subplot(2, 1, 2);
    analyze_spectrum(signal, fs);
    
    fprintf('Custom signal generated and analyzed.\n');
end

function compare_filter_methods_demo()
    fprintf('\n--- Filter Methods Comparison ---\n');
    
    % Generate test signal
    fs = 1000;
    t = 0:1/fs:2;
    signal = sin(2*pi*50*t) + 0.5*sin(2*pi*100*t) + 0.2*randn(size(t));
    
    compare_filter_methods(signal, 75);
    
    fprintf('Filter methods comparison complete.\n');
end

function window_effects_demo()
    fprintf('\n--- Window Function Effects ---\n');
    
    % Generate test signal
    fs = 1000;
    t = 0:1/fs:1;
    signal = sin(2*pi*50*t) + 0.3*sin(2*pi*120*t) + 0.1*randn(size(t));
    
    compare_windows(signal, fs);
    
    fprintf('Window function effects demonstration complete.\n');
end