% Location: mini_projects/signal_processing_simulation/data/generate_sample_signals.m
% Generate Sample Signal Data for Signal Processing Demonstrations

function generate_sample_signals()
    % Generate comprehensive sample signals for demonstrations
    
    fprintf('Generating sample signal data...\n');
    
    % Create data directory if it doesn't exist
    data_dir = fileparts(mfilename('fullpath'));
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    % Generate different sample signals
    generate_ecg_sample(data_dir);
    generate_speech_sample(data_dir);
    generate_noisy_sine(data_dir);
    generate_chirp_signals(data_dir);
    generate_multi_tone(data_dir);
    generate_biomedical_signals(data_dir);
    
    fprintf('Sample signal generation complete!\n');
    fprintf('Generated files in: %s\n', data_dir);
end

function generate_ecg_sample(data_dir)
    % Generate synthetic ECG signal
    
    fprintf('  Generating ECG sample...\n');
    
    fs = 500; % 500 Hz sampling rate (typical for ECG)
    duration = 10; % 10 seconds
    t = 0:1/fs:duration-1/fs;
    
    % ECG parameters
    heart_rate = 72; % BPM
    rr_interval = 60 / heart_rate; % seconds between beats
    
    % Generate ECG waveform
    ecg_signal = zeros(size(t));
    beat_times = 0:rr_interval:duration;
    
    for beat_time = beat_times
        if beat_time < duration - 1
            % Find indices for this beat
            beat_start = round(beat_time * fs) + 1;
            beat_duration = round(0.8 * fs); % 0.8 second beat duration
            beat_end = min(beat_start + beat_duration - 1, length(t));
            
            if beat_end <= length(t)
                % Generate PQRST complex
                beat_t = (0:beat_end-beat_start) / fs;
                
                % P wave (0.1s duration, starts at 0.05s)
                p_wave = 0.1 * exp(-((beat_t - 0.05) / 0.03).^2);
                
                % QRS complex (0.1s duration, starts at 0.15s) 
                qrs_t = beat_t - 0.15;
                q_wave = -0.05 * exp(-(qrs_t / 0.01).^2);
                r_wave = 0.8 * exp(-(qrs_t / 0.02).^2);
                s_wave = -0.1 * exp(-((qrs_t - 0.02) / 0.01).^2);
                qrs_complex = q_wave + r_wave + s_wave;
                
                % T wave (0.2s duration, starts at 0.4s)
                t_wave = 0.2 * exp(-((beat_t - 0.4) / 0.05).^2);
                
                % Combine waves
                beat_signal = p_wave + qrs_complex + t_wave;
                
                % Add to main signal
                ecg_signal(beat_start:beat_end) = beat_signal;
            end
        end
    end
    
    % Add realistic noise
    noise_power = 0.02;
    ecg_signal = ecg_signal + noise_power * randn(size(ecg_signal));
    
    % Add baseline wander (breathing artifact)
    baseline_freq = 0.3; % Hz
    baseline_drift = 0.05 * sin(2*pi*baseline_freq*t);
    ecg_signal = ecg_signal + baseline_drift;
    
    % Add power line interference
    powerline_noise = 0.01 * sin(2*pi*60*t); % 60 Hz
    ecg_signal = ecg_signal + powerline_noise;
    
    % Save data
    signal_data.signal = ecg_signal;
    signal_data.time = t;
    signal_data.fs = fs;
    signal_data.description = 'Synthetic ECG signal with noise and artifacts';
    signal_data.units = 'mV';
    signal_data.heart_rate = heart_rate;
    
    save(fullfile(data_dir, 'ecg_sample.mat'), 'signal_data');
end

function generate_speech_sample(data_dir)
    % Generate synthetic speech-like signal
    
    fprintf('  Generating speech sample...\n');
    
    fs = 8000; % 8 kHz (telephone quality)
    duration = 3; % 3 seconds
    t = 0:1/fs:duration-1/fs;
    
    % Generate formant-based speech synthesis
    speech_signal = zeros(size(t));
    
    % Vowel sounds with different formant frequencies
    vowel_segments = [
        struct('duration', 0.5, 'f1', 730, 'f2', 1090, 'f3', 2440); % /a/
        struct('duration', 0.5, 'f1', 270, 'f2', 2290, 'f3', 3010); % /i/
        struct('duration', 0.5, 'f1', 300, 'f2', 870, 'f3', 2240);  % /u/
        struct('duration', 0.5, 'f1', 530, 'f2', 1840, 'f3', 2480); % /e/
        struct('duration', 0.5, 'f1', 660, 'f2', 1720, 'f3', 2410); % /æ/
        struct('duration', 0.5, 'f1', 490, 'f2', 1350, 'f3', 1690); % /o/
    ];
    
    current_time = 0;
    
    for i = 1:length(vowel_segments)
        vowel = vowel_segments(i);
        if current_time + vowel.duration <= duration
            % Time indices for this vowel
            start_idx = round(current_time * fs) + 1;
            end_idx = round((current_time + vowel.duration) * fs);
            
            if end_idx <= length(t)
                vowel_t = t(start_idx:end_idx);
                
                % Fundamental frequency (pitch) with variation
                f0 = 120 + 20 * sin(2*pi*2*vowel_t); % Varying pitch
                
                % Generate harmonics
                harmonic_signal = zeros(size(vowel_t));
                
                for harmonic = 1:10
                    harmonic_freq = harmonic * f0;
                    % Formant filtering effect (simplified)
                    amplitude = exp(-0.1*harmonic) .* ...
                               (exp(-((harmonic_freq - vowel.f1)/100).^2) + ...
                                0.7*exp(-((harmonic_freq - vowel.f2)/200).^2) + ...
                                0.5*exp(-((harmonic_freq - vowel.f3)/300).^2));
                    
                    harmonic_signal = harmonic_signal + amplitude .* sin(2*pi*harmonic_freq.*vowel_t);
                end
                
                % Add to main signal
                speech_signal(start_idx:end_idx) = harmonic_signal;
                current_time = current_time + vowel.duration;
            end
        end
    end
    
    % Add consonant-like noise bursts
    for i = 1:5
        burst_time = rand() * duration;
        burst_idx = round(burst_time * fs);
        burst_duration = round(0.05 * fs); % 50ms burst
        
        if burst_idx + burst_duration <= length(speech_signal)
            noise_burst = 0.1 * randn(1, burst_duration);
            speech_signal(burst_idx:burst_idx+burst_duration-1) = ...
                speech_signal(burst_idx:burst_idx+burst_duration-1) + noise_burst;
        end
    end
    
    % Normalize
    if max(abs(speech_signal)) > 0
        speech_signal = speech_signal / max(abs(speech_signal)) * 0.8;
    end
    
    % Save data
    signal_data.signal = speech_signal;
    signal_data.time = t;
    signal_data.fs = fs;
    signal_data.description = 'Synthetic speech signal with formants';
    signal_data.units = 'normalized amplitude';
    signal_data.vowel_info = vowel_segments;
    
    save(fullfile(data_dir, 'speech_sample.mat'), 'signal_data');
end

function generate_noisy_sine(data_dir)
    % Generate sine waves with different types of noise
    
    fprintf('  Generating noisy sine waves...\n');
    
    fs = 1000;
    duration = 5;
    t = 0:1/fs:duration-1/fs;
    
    % Base sine wave
    frequency = 50; % Hz
    clean_sine = sin(2*pi*frequency*t);
    
    % Different noise types
    awgn_10db = clean_sine + 10^(-10/20) * randn(size(clean_sine)); % 10dB SNR
    awgn_20db = clean_sine + 10^(-20/20) * randn(size(clean_sine)); % 20dB SNR
    
    % Impulse noise
    impulse_noise = clean_sine;
    impulse_locations = rand(size(clean_sine)) < 0.01; % 1% impulse probability
    impulse_noise(impulse_locations) = impulse_noise(impulse_locations) + 2*(rand(1,sum(impulse_locations))-0.5);
    
    % Colored noise
    white_noise = randn(size(clean_sine));
    % Simple colored noise (pink noise approximation)
    colored_noise = filter([1 0.5 0.25], [1 -0.5 0.25], white_noise);
    colored_noise = colored_noise / std(colored_noise) * 0.2;
    colored_sine = clean_sine + colored_noise;
    
    % Interference
    interference_freq = 120; % Hz (power line frequency)
    interference = clean_sine + 0.3*sin(2*pi*interference_freq*t);
    
    % Save all variants
    signal_data.clean = clean_sine;
    signal_data.awgn_10db = awgn_10db;
    signal_data.awgn_20db = awgn_20db;
    signal_data.impulse_noise = impulse_noise;
    signal_data.colored_noise = colored_sine;
    signal_data.interference = interference;
    signal_data.time = t;
    signal_data.fs = fs;
    signal_data.frequency = frequency;
    signal_data.description = 'Sine wave with various noise types';
    
    save(fullfile(data_dir, 'noisy_sine.mat'), 'signal_data');
end

function generate_chirp_signals(data_dir)
    % Generate various chirp (frequency sweep) signals
    
    fprintf('  Generating chirp signals...\n');
    
    fs = 2000;
    duration = 4;
    t = 0:1/fs:duration-1/fs;
    
    % Linear chirp (frequency increases linearly)
    f0_linear = 10; % Start frequency
    f1_linear = 200; % End frequency
    linear_chirp = chirp(t, f0_linear, duration, f1_linear, 'linear');
    
    % Logarithmic chirp (frequency increases exponentially)
    f0_log = 10;
    f1_log = 200;
    log_chirp = chirp(t, f0_log, duration, f1_log, 'logarithmic');
    
    % Quadratic chirp
    f0_quad = 20;
    f1_quad = 100;
    quad_chirp = chirp(t, f0_quad, duration, f1_quad, 'quadratic');
    
    % Multi-directional chirp (up then down)
    mid_time = duration / 2;
    up_indices = t <= mid_time;
    down_indices = t > mid_time;
    up_chirp = chirp(t(up_indices), 50, mid_time, 150, 'linear');
    down_chirp = chirp(t(down_indices) - mid_time, 150, mid_time, 50, 'linear');
    bidirectional_chirp = zeros(size(t));
    bidirectional_chirp(up_indices) = up_chirp;
    bidirectional_chirp(down_indices) = down_chirp;
    
    % Frequency modulated chirp
    carrier_freq = 100;
    modulation_freq = 2;
    fm_chirp = sin(2*pi*carrier_freq*t + 50*sin(2*pi*modulation_freq*t));
    
    % Save data
    signal_data.linear_chirp = linear_chirp;
    signal_data.log_chirp = log_chirp;
    signal_data.quad_chirp = quad_chirp;
    signal_data.bidirectional_chirp = bidirectional_chirp;
    signal_data.fm_chirp = fm_chirp;
    signal_data.time = t;
    signal_data.fs = fs;
    signal_data.description = 'Various chirp signal types';
    signal_data.parameters = struct(...
        'linear', [f0_linear, f1_linear], ...
        'log', [f0_log, f1_log], ...
        'quad', [f0_quad, f1_quad]);
    
    save(fullfile(data_dir, 'chirp_signals.mat'), 'signal_data');
end

function generate_multi_tone(data_dir)
    % Generate multi-frequency composite signals
    
    fprintf('  Generating multi-tone signals...\n');
    
    fs = 2000;
    duration = 3;
    t = 0:1/fs:duration-1/fs;
    
    % Simple multi-tone (3 frequencies)
    frequencies_simple = [25, 50, 100];
    amplitudes_simple = [1, 0.8, 0.6];
    simple_multitone = zeros(size(t));
    
    for i = 1:length(frequencies_simple)
        simple_multitone = simple_multitone + amplitudes_simple(i) * sin(2*pi*frequencies_simple(i)*t);
    end
    
    % Complex multi-tone (10 frequencies)
    frequencies_complex = 20:20:200; % 20, 40, 60, ..., 200 Hz
    amplitudes_complex = 1 ./ sqrt(frequencies_complex); % 1/f amplitude characteristic
    complex_multitone = zeros(size(t));
    
    for i = 1:length(frequencies_complex)
        phase = 2*pi*rand(); % Random phase
        complex_multitone = complex_multitone + amplitudes_complex(i) * sin(2*pi*frequencies_complex(i)*t + phase);
    end
    
    % Musical chord (major triad)
    fundamental = 220; % A3 note
    major_third = fundamental * 5/4; % C#
    perfect_fifth = fundamental * 3/2; % E
    musical_chord = sin(2*pi*fundamental*t) + 0.8*sin(2*pi*major_third*t) + 0.6*sin(2*pi*perfect_fifth*t);
    
    % Amplitude modulated multi-tone
    modulation_rate = 2; % Hz
    am_envelope = 0.5 * (1 + sin(2*pi*modulation_rate*t));
    am_multitone = simple_multitone .* am_envelope;
    
    % Beat frequencies (two close frequencies)
    f1_beat = 100;
    f2_beat = 102; % 2 Hz beat frequency
    beat_signal = sin(2*pi*f1_beat*t) + sin(2*pi*f2_beat*t);
    
    % Save data
    signal_data.simple_multitone = simple_multitone;
    signal_data.complex_multitone = complex_multitone;
    signal_data.musical_chord = musical_chord;
    signal_data.am_multitone = am_multitone;
    signal_data.beat_signal = beat_signal;
    signal_data.time = t;
    signal_data.fs = fs;
    signal_data.description = 'Multi-frequency composite signals';
    signal_data.frequency_info = struct(...
        'simple_frequencies', frequencies_simple, ...
        'complex_frequencies', frequencies_complex, ...
        'musical_notes', [fundamental, major_third, perfect_fifth]);
    
    save(fullfile(data_dir, 'multi_tone.mat'), 'signal_data');
end

function generate_biomedical_signals(data_dir)
    % Generate additional biomedical signal examples
    
    fprintf('  Generating biomedical signals...\n');
    
    % EEG-like signal
    fs_eeg = 256;
    duration_eeg = 10;
    t_eeg = 0:1/fs_eeg:duration_eeg-1/fs_eeg;
    
    % EEG frequency bands
    delta_band = 0.5 * sin(2*pi*2*t_eeg + 2*pi*rand()); % Delta (1-4 Hz)
    theta_band = 0.3 * sin(2*pi*6*t_eeg + 2*pi*rand()); % Theta (4-8 Hz)
    alpha_band = 0.8 * sin(2*pi*10*t_eeg + 2*pi*rand()); % Alpha (8-13 Hz)
    beta_band = 0.2 * sin(2*pi*20*t_eeg + 2*pi*rand()); % Beta (13-30 Hz)
    
    eeg_signal = delta_band + theta_band + alpha_band + beta_band + 0.1*randn(size(t_eeg));
    
    % EMG-like signal (muscle activity)
    fs_emg = 1000;
    duration_emg = 5;
    t_emg = 0:1/fs_emg:duration_emg-1/fs_emg;
    
    % Muscle activation periods
    emg_signal = 0.05 * randn(size(t_emg)); % Baseline noise
    
    % Add muscle activation bursts
    activation_times = [1, 2.5, 4]; % seconds
    for i = 1:length(activation_times)
        activation_time = activation_times(i);
        start_idx = round(activation_time * fs_emg);
        burst_duration = round(0.5 * fs_emg); % 0.5 second burst
        end_idx = min(start_idx + burst_duration, length(emg_signal));
        
        if start_idx <= length(emg_signal) && start_idx > 0
            % High-frequency burst with envelope
            burst_indices = start_idx:end_idx;
            envelope = exp(-5 * (burst_indices - start_idx) / fs_emg); % Exponential decay
            burst_signal = envelope .* (0.5 * randn(size(burst_indices)));
            emg_signal(burst_indices) = emg_signal(burst_indices) + burst_signal;
        end
    end
    
    % Blood pressure-like signal (arterial pulse)
    fs_bp = 125;
    duration_bp = 10;
    t_bp = 0:1/fs_bp:duration_bp-1/fs_bp;
    
    % Generate arterial pulse waveform
    pulse_rate = 75; % BPM
    pulse_interval = 60 / pulse_rate; % seconds between pulses
    bp_signal = zeros(size(t_bp));
    
    pulse_times = 0:pulse_interval:duration_bp;
    for pulse_time = pulse_times
        if pulse_time < duration_bp - 0.5
            pulse_start = round(pulse_time * fs_bp) + 1;
            pulse_duration = round(0.4 * fs_bp); % 0.4 second pulse duration
            pulse_end = min(pulse_start + pulse_duration - 1, length(t_bp));
            
            if pulse_end <= length(t_bp)
                pulse_t = (0:pulse_end-pulse_start) / fs_bp;
                
                % Systolic peak followed by diastolic decay
                systolic_peak = exp(-((pulse_t - 0.1) / 0.05).^2);
                diastolic_decay = 0.3 * exp(-pulse_t / 0.15);
                
                % Combine systolic and diastolic components
                pulse_waveform = systolic_peak + diastolic_decay;
                
                % Add to main signal
                bp_signal(pulse_start:pulse_end) = pulse_waveform;
            end
        end
    end
    
    % Add baseline pressure and noise
    baseline_pressure = 80; % mmHg (diastolic baseline)
    systolic_amplitude = 40; % mmHg (systolic - diastolic)
    bp_signal = baseline_pressure + systolic_amplitude * bp_signal + 2 * randn(size(bp_signal));
    
    % Respiratory signal
    fs_resp = 25;
    duration_resp = 60; % 1 minute
    t_resp = 0:1/fs_resp:duration_resp-1/fs_resp;
    
    resp_rate = 15; % breaths per minute
    resp_freq = resp_rate / 60; % Hz
    
    % Generate breathing pattern with natural variation
    resp_signal = sin(2*pi*resp_freq*t_resp) + ...
                  0.3*sin(2*pi*2*resp_freq*t_resp) + ... % Harmonic
                  0.1*randn(size(t_resp)); % Noise
    
    % Add slow variation in breathing depth
    depth_variation = 0.2 * sin(2*pi*0.05*t_resp); % 0.05 Hz variation
    resp_signal = resp_signal .* (1 + depth_variation);
    
    % Save biomedical signals
    bio_data.eeg_signal = eeg_signal;
    bio_data.eeg_time = t_eeg;
    bio_data.eeg_fs = fs_eeg;
    bio_data.emg_signal = emg_signal;
    bio_data.emg_time = t_emg;
    bio_data.emg_fs = fs_emg;
    bio_data.bp_signal = bp_signal;
    bio_data.bp_time = t_bp;
    bio_data.bp_fs = fs_bp;
    bio_data.resp_signal = resp_signal;
    bio_data.resp_time = t_resp;
    bio_data.resp_fs = fs_resp;
    bio_data.description = 'Synthetic biomedical signals (EEG, EMG, BP, Respiratory)';
    bio_data.signal_info = struct(...
        'eeg_bands', struct('delta', '1-4Hz', 'theta', '4-8Hz', 'alpha', '8-13Hz', 'beta', '13-30Hz'), ...
        'pulse_rate_bpm', pulse_rate, ...
        'resp_rate_bpm', resp_rate);
    
    save(fullfile(data_dir, 'biomedical_signals.mat'), 'bio_data');
    
    fprintf('  Biomedical signals generation complete.\n');
end