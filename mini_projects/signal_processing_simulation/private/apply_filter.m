% Location: mini_projects/signal_processing_simulation/private/apply_filter.m
% (split out of filter_design.m so it can be called from other files)

function [filtered_signal, filter_coeffs] = apply_filter(signal, filter_type, cutoff_freq, fs, varargin)
    % Apply digital filters to signals
    %
    % Inputs:
    %   signal - input signal
    %   filter_type - 'lowpass', 'highpass', 'bandpass', 'bandstop'
    %   cutoff_freq - cutoff frequency (Hz) or [low, high] for bandpass/bandstop
    %   fs - sampling frequency (Hz)
    %   varargin - additional parameters (order, method)
    %
    % Outputs:
    %   filtered_signal - filtered output
    %   filter_coeffs - filter coefficients [b, a]
    
    % Default parameters
    filter_order = 4;
    method = 'butter'; % 'butter', 'cheby1', 'cheby2', 'ellip'
    
    if length(varargin) >= 1
        filter_order = varargin{1};
    end
    if length(varargin) >= 2
        method = varargin{2};
    end
    
    % Normalize cutoff frequencies
    nyquist = fs / 2;
    
    switch lower(filter_type)
        case 'lowpass'
            Wn = cutoff_freq / nyquist;
            if Wn >= 1
                warning('Cutoff frequency too high, setting to 0.9*Nyquist');
                Wn = 0.9;
            end
            
        case 'highpass'
            Wn = cutoff_freq / nyquist;
            if Wn >= 1
                warning('Cutoff frequency too high, setting to 0.9*Nyquist');
                Wn = 0.9;
            end
            
        case {'bandpass', 'bandstop'}
            if length(cutoff_freq) ~= 2
                error('Bandpass/bandstop filters require [low_freq, high_freq]');
            end
            Wn = cutoff_freq / nyquist;
            if any(Wn >= 1)
                warning('Cutoff frequencies too high, adjusting');
                Wn = min(Wn, 0.9);
            end
            
        otherwise
            error('Unknown filter type: %s', filter_type);
    end
    
    % Design filter based on method
    switch lower(method)
        case 'butter'
            [b, a] = butter(filter_order, Wn, filter_type);
            
        case 'cheby1'
            ripple = 1; % dB
            if length(varargin) >= 3
                ripple = varargin{3};
            end
            [b, a] = cheby1(filter_order, ripple, Wn, filter_type);
            
        case 'cheby2'
            stopband_atten = 40; % dB
            if length(varargin) >= 3
                stopband_atten = varargin{3};
            end
            [b, a] = cheby2(filter_order, stopband_atten, Wn, filter_type);
            
        case 'ellip'
            passband_ripple = 1; % dB
            stopband_atten = 40; % dB
            if length(varargin) >= 3
                passband_ripple = varargin{3};
            end
            if length(varargin) >= 4
                stopband_atten = varargin{4};
            end
            [b, a] = ellip(filter_order, passband_ripple, stopband_atten, Wn, filter_type);
            
        otherwise
            error('Unknown filter method: %s', method);
    end
    
    % Apply filter
    filtered_signal = filtfilt(b, a, signal);
    filter_coeffs = {b, a};
end
