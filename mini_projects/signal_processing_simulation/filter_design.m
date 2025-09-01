% Location: mini_projects/signal_processing_simulation/filter_design.m
% Digital Filter Design and Application

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

function plot_filter_response(b, a, fs)
    % Plot frequency response of filter
    
    [h, w] = freqz(b, a, 1024, fs);
    
    figure;
    subplot(2, 1, 1);
    semilogx(w, 20*log10(abs(h)));
    title('Filter Magnitude Response');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    
    subplot(2, 1, 2);
    semilogx(w, angle(h)*180/pi);
    title('Filter Phase Response');
    xlabel('Frequency (Hz)');
    ylabel('Phase (degrees)');
    grid on;
end

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