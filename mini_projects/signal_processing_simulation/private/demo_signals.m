% Location: mini_projects/signal_processing_simulation/private/demo_signals.m
% (split out of signal_generator.m so it can be called from other files)

function demo_signals()
    % Demonstrate different signal types
    fs = 1000; % sampling frequency
    duration = 2; % seconds
    frequency = 50; % Hz
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Sine wave
    subplot(3, 2, 1);
    [t, sine_sig] = generate_signal('sine', fs, duration, frequency, 1);
    plot(t(1:500), sine_sig(1:500));
    title('Sine Wave');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Square wave
    subplot(3, 2, 2);
    [t, square_sig] = generate_signal('square', fs, duration, frequency, 1, 30);
    plot(t(1:500), square_sig(1:500));
    title('Square Wave (30% duty cycle)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Sawtooth wave
    subplot(3, 2, 3);
    [t, saw_sig] = generate_signal('sawtooth', fs, duration, frequency, 1);
    plot(t(1:500), saw_sig(1:500));
    title('Sawtooth Wave');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % White noise
    subplot(3, 2, 4);
    [t, noise_sig] = generate_signal('noise', fs, duration, frequency, 1, 'white');
    plot(t(1:500), noise_sig(1:500));
    title('White Noise');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Chirp signal
    subplot(3, 2, 5);
    [t, chirp_sig] = generate_signal('chirp', fs, duration, 10, 1, 100);
    plot(t, chirp_sig);
    title('Chirp Signal (10-100 Hz)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Composite signal
    subplot(3, 2, 6);
    freqs = [20, 50, 80];
    amps = [1, 0.7, 0.4];
    [t, comp_sig] = generate_signal('composite', fs, duration, 20, 1, freqs, amps);
    plot(t(1:500), comp_sig(1:500));
    title('Composite Signal (20, 50, 80 Hz)');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    sgtitle('Signal Generator Demonstration');
end
