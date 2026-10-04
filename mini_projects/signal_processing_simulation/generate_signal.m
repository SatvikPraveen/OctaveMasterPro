% Location: mini_projects/signal_processing_simulation/generate_signal.m
% (split out of signal_generator.m so it can be called from other files)

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
