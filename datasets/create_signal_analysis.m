% File location: OctaveMasterPro/datasets/create_signal_analysis.m
% Script to generate signal_analysis.mat for signal processing learning

fprintf('Generating signal analysis data...\n');

% Sampling parameters
fs = 1000; % Sampling frequency (Hz)
N = 5000;  % Number of samples
t = (0:N-1) / fs; % Time vector

% Generate clean signal - composite of multiple sinusoids
fprintf('Creating clean signal...\n');
f1 = 50;   % 50 Hz component
f2 = 120;  % 120 Hz component  
f3 = 200;  % 200 Hz component

clean_signal = 2.0 * sin(2*pi*f1*t) + ...
               1.5 * sin(2*pi*f2*t + pi/4) + ...
               1.0 * sin(2*pi*f3*t + pi/2) + ...
               0.5 * sin(2*pi*5*t); % Low frequency component

% Add realistic noise to create noisy signal
fprintf('Adding noise...\n');
SNR_dB = 20; % Signal-to-noise ratio in dB
signal_power = mean(clean_signal.^2);
noise_power = signal_power / (10^(SNR_dB/10));
noise = sqrt(noise_power) * randn(size(clean_signal));

noisy_signal = clean_signal + noise;

% Create filtered signals using different filter types
fprintf('Applying filters...\n');

% Design filters
% 1. Low-pass filter (cutoff at 100 Hz)
[b_lp, a_lp] = butter(4, 100/(fs/2), 'low');
filtered_lowpass = filter(b_lp, a_lp, noisy_signal);

% 2. Band-pass filter (40-60 Hz to isolate 50 Hz component)
[b_bp, a_bp] = butter(4, [40 60]/(fs/2), 'bandpass');
filtered_bandpass = filter(b_bp, a_bp, noisy_signal);

% 3. High-pass filter (cutoff at 80 Hz)
[b_hp, a_hp] = butter(4, 80/(fs/2), 'high');
filtered_highpass = filter(b_hp, a_hp, noisy_signal);

% Combine filtered signals into matrix
filtered_signals = [filtered_lowpass; filtered_bandpass; filtered_highpass];

% Perform frequency analysis
fprintf('Computing frequency analysis...\n');

% FFT of signals
fft_clean = fft(clean_signal);
fft_noisy = fft(noisy_signal);
fft_filtered = fft(filtered_signals, [], 2);

% Frequency vector
frequencies = (0:N-1) * fs / N;

% Power spectral density
psd_clean = abs(fft_clean).^2 / (fs * N);
psd_noisy = abs(fft_noisy).^2 / (fs * N);
psd_filtered = abs(fft_filtered).^2 / (fs * N);

% Create frequency analysis structure
frequency_data.frequencies = frequencies;
frequency_data.fft_clean = fft_clean;
frequency_data.fft_noisy = fft_noisy;
frequency_data.fft_filtered = fft_filtered;
frequency_data.psd_clean = psd_clean;
frequency_data.psd_noisy = psd_noisy;
frequency_data.psd_filtered = psd_filtered;
frequency_data.sampling_rate = fs;

% Generate additional signal types for learning
fprintf('Creating additional signal types...\n');

% Chirp signal (frequency sweep)
f_start = 10;  % Start frequency
f_end = 400;   % End frequency
chirp_signal = chirp(t, f_start, max(t), f_end, 'linear');

% Square wave
square_signal = square(2*pi*25*t); % 25 Hz square wave

% Sawtooth wave
sawtooth_signal = sawtooth(2*pi*15*t); % 15 Hz sawtooth

% Pulse train
pulse_width = 0.1; % 10% duty cycle
pulse_signal = pulstran(t, 0:0.1:max(t), 'rectpuls', pulse_width);

% Amplitude modulated signal
carrier_freq = 100;  % Carrier frequency
mod_freq = 5;        % Modulation frequency
am_signal = (1 + 0.5*cos(2*pi*mod_freq*t)) .* cos(2*pi*carrier_freq*t);

% Frequency modulated signal
freq_deviation = 20; % Frequency deviation
fm_signal = cos(2*pi*carrier_freq*t + (freq_deviation/mod_freq)*sin(2*pi*mod_freq*t));

% Create time-frequency analysis data
fprintf('Computing spectrograms...\n');

% Parameters for spectrogram
window_length = 256;
overlap = 128;
nfft = 512;

% Compute spectrograms (simplified version)
[S_clean, f_spec, t_spec] = specgram(clean_signal, nfft, fs, window_length, overlap);
[S_noisy, ~, ~] = specgram(noisy_signal, nfft, fs, window_length, overlap);
[S_chirp, ~, ~] = specgram(chirp_signal, nfft, fs, window_length, overlap);

% Create spectrogram structure
spectrogram_data.S_clean = S_clean;
spectrogram_data.S_noisy = S_noisy;
spectrogram_data.S_chirp = S_chirp;
spectrogram_data.frequencies = f_spec;
spectrogram_data.time = t_spec;

% Filter coefficients for reference
filter_coeffs.lowpass.b = b_lp;
filter_coeffs.lowpass.a = a_lp;
filter_coeffs.bandpass.b = b_bp;
filter_coeffs.bandpass.a = a_bp;
filter_coeffs.highpass.b = b_hp;
filter_coeffs.highpass.a = a_hp;

% Signal parameters for reference
signal_params.sampling_rate = fs;
signal_params.duration = max(t);
signal_params.frequencies = [f1, f2, f3];
signal_params.amplitudes = [2.0, 1.5, 1.0];
signal_params.snr_db = SNR_dB;

% Save all data to MAT file
fprintf('Saving to signal_analysis.mat...\n');
save('signal_analysis.mat', ...
     'clean_signal', 'noisy_signal', 'filtered_signals', ...
     'chirp_signal', 'square_signal', 'sawtooth_signal', 'pulse_signal', ...
     'am_signal', 'fm_signal', ...
     'frequency_data', 'spectrogram_data', 'filter_coeffs', 'signal_params', ...
     't', 'fs', '-v7');

fprintf('Successfully created signal_analysis.mat\n');
fprintf('Contains:\n');
fprintf('  - clean_signal: Pure multi-frequency sinusoid\n');
fprintf('  - noisy_signal: Signal + noise (SNR = %d dB)\n', SNR_dB);
fprintf('  - filtered_signals: 3x%d matrix (LP, BP, HP filters)\n', length(filtered_signals));
fprintf('  - Various signal types: chirp, square, sawtooth, pulse, AM, FM\n');
fprintf('  - frequency_data: FFT and PSD analysis\n');
fprintf('  - spectrogram_data: Time-frequency analysis\n');
fprintf('  - filter_coeffs: Filter design coefficients\n');
fprintf('  - signal_params: Generation parameters\n');