% Location: mini_projects/signal_processing_simulation/signal_generator.m
% Signal Generator - Generate various types of signals for processing

function [t, signal] = generate_signal(signal_type, fs, duration, frequency, amplitude, varargin)
    % Generate different types of signals
    % 
    % Inputs:
    %   signal_type - 'sine', 'square', 'sawtooth', 'noise', 'chirp'
    %   fs - sampling frequency (Hz)
    %   duration - signal duration (seconds)
    %   frequency - signal frequency (Hz)
    %   amplitude - signal amplitude
    %   varargin - additional parameters for specific signals
    %
    % Outputs:
    %   t - time vector
    %   signal - generated signal
    
    if nargin < 5
        amplitude = 1;
    end
    
    % Create time vector
    t = 0:1/fs:(duration - 1/fs);
    N = length(t);
    
    switch lower(signal_type)
        case 'sine'
            phase = 0;
            if length(varargin) >= 1
                phase = varargin{1};
            end
            signal = amplitude * sin(2*pi*frequency*t + phase);
            
        case 'square'
            duty_cycle = 50; % percent
            if length(varargin) >= 1
                duty_cycle = varargin{1};
            end
            signal = amplitude * square(2*pi*frequency*t, duty_cycle);
            
        case 'sawtooth'
            width = 1; % 1 for sawtooth, 0 for reverse sawtooth
            if length(varargin) >= 1
                width = varargin{1};
            end
            signal = amplitude * sawtooth(2*pi*frequency*t, width);
            
        case 'noise'
            noise_type = 'white'; % 'white', 'pink'
            if length(varargin) >= 1
                noise_type = varargin{1};
            end
            if strcmp(noise_type, 'white')
                signal = amplitude * randn(1, N);
            else
                % Simple pink noise approximation
                white_noise = randn(1, N);
                signal = amplitude * filter([1 -0.99], 1, white_noise);
            end
            
        case 'chirp'
            f_end = frequency * 2; % end frequency
            if length(varargin) >= 1
                f_end = varargin{1};
            end
            signal = amplitude * chirp(t, frequency, duration, f_end);
            
        case 'composite'
            % Multi-frequency signal
            frequencies = [frequency, frequency*2, frequency*3];
            amplitudes = [amplitude, amplitude*0.5, amplitude*0.25];
            if length(varargin) >= 1
                frequencies = varargin{1};
            end
            if length(varargin) >= 2
                amplitudes = varargin{2};
            end
            
            signal = zeros(size(t));
            for i = 1:length(frequencies)
                if i <= length(amplitudes)
                    amp = amplitudes(i);
                else
                    amp = amplitude / i;
                end
                signal = signal + amp * sin(2*pi*frequencies(i)*t);
            end
            
        otherwise
            error('Unknown signal type: %s', signal_type);
    end
    
    % Add noise if specified
    if length(varargin) >= 3 && varargin{3} > 0
        snr_db = varargin{3};
        noise_power = var(signal) / (10^(snr_db/10));
        signal = signal + sqrt(noise_power) * randn(size(signal));
    end
end

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