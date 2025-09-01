% File location: OctaveMasterPro/datasets/create_signal_files.m
% Script to generate signal files for signal processing learning

fprintf('Creating signal files for OctaveMasterPro...\n');

% Create signal directories
if ~exist('signals', 'dir')
    mkdir('signals');
end
if ~exist('signals/audio', 'dir')
    mkdir('signals/audio');
end
if ~exist('signals/raw', 'dir')
    mkdir('signals/raw');
end
if ~exist('signals/synthetic', 'dir')
    mkdir('signals/synthetic');
end

%% Audio Signals
fprintf('Creating audio signals...\n');

% Sampling rate for audio
fs_audio = 44100; % Standard audio sampling rate
duration = 3; % 3 seconds
t_audio = (0:1/fs_audio:duration-1/fs_audio)';

% Pure sine wave at 440 Hz (A4 note)
sine_440 = sin(2*pi*440*t_audio);
audiowrite('signals/audio/sine_wave_440hz.wav', sine_440, fs_audio);

% Chirp signal (frequency sweep from 100 Hz to 2000 Hz)
chirp_signal = chirp(t_audio, 100, duration, 2000, 'linear');
audiowrite('signals/audio/chirp_signal.wav', chirp_signal, fs_audio);

% Musical chord (C major: C-E-G)
freq_C = 261.63; % C4
freq_E = 329.63; % E4
freq_G = 392.00; % G4

chord = 0.33 * (sin(2*pi*freq_C*t_audio) + ...
                sin(2*pi*freq_E*t_audio) + ...
                sin(2*pi*freq_G*t_audio));

% Add envelope to make it sound more natural
envelope = exp(-t_audio/1.5); % Exponential decay
chord = chord .* envelope;

audiowrite('signals/audio/music_sample.wav', chord, fs_audio);

% Speech with noise simulation
% Generate speech-like signal using multiple frequencies
speech_freqs = [200, 400, 800, 1600, 3200]; % Formant-like frequencies
speech_signal = zeros(size(t_audio));

for i = 1:length(speech_freqs)
    % Modulate amplitude randomly to simulate speech patterns
    modulation = 0.5 + 0.5 * sin(2*pi*t_audio*2*(i+1)) .* (0.8 + 0.4*randn(size(t_audio)));
    speech_signal = speech_signal + 0.2 * sin(2*pi*speech_freqs(i)*t_audio) .* modulation;
end

% Add background noise
noise_level = 0.1;
background_noise = noise_level * randn(size(speech_signal));
noisy_speech = speech_signal + background_noise;

% Normalize
noisy_speech = noisy_speech / max(abs(noisy_speech)) * 0.8;
audiowrite('signals/audio/noisy_speech.wav', noisy_speech, fs_audio);

%% Raw Signal Data Files
fprintf('Creating raw signal data files...\n');

% ECG signal simulation
fs_ecg = 360; % Common ECG sampling rate
duration_ecg = 10; % 10 seconds
t_ecg = (0:1/fs_ecg:duration_ecg-1/fs_ecg)';

% Simulate ECG waveform (simplified)
heart_rate = 72; % beats per minute
beat_period = 60 / heart_rate; % seconds per beat

ecg_signal = zeros(size(t_ecg));

% Generate individual heartbeats
for beat = 0:beat_period:duration_ecg
    % P wave, QRS complex, T wave simulation
    beat_time = t_ecg - beat;
    
    % QRS complex (main peak)
    qrs_mask = abs(beat_time) < 0.05;
    ecg_signal(qrs_mask) = ecg_signal(qrs_mask) + exp(-beat_time(qrs_mask).^2/0.001);
    
    % P wave
    p_mask = abs(beat_time + 0.15) < 0.03;
    ecg_signal(p_mask) = ecg_signal(p_mask) + 0.3 * exp(-(beat_time(p_mask) + 0.15).^2/0.0005);
    
    % T wave
    t_mask = abs(beat_time - 0.2) < 0.08;
    ecg_signal(t_mask) = ecg_signal(t_mask) + 0.4 * exp(-(beat_time(t_mask) - 0.2).^2/0.002);
end

% Add baseline wander and noise
baseline_wander = 0.1 * sin(2*pi*0.5*t_ecg);
ecg_noise = 0.05 * randn(size(ecg_signal));
ecg_signal = ecg_signal + baseline_wander + ecg_noise;

% Save as ASCII data
save('signals/raw/ecg_signal.dat', 'ecg_signal', '-ascii');

% 3-axis accelerometer data
fs_accel = 100; % 100 Hz sampling
duration_accel = 30; % 30 seconds
t_accel = (0:1/fs_accel:duration_accel-1/fs_accel)';

% Simulate walking motion
step_freq = 1.5; % 1.5 steps per second

% X-axis: forward-backward sway
accel_x = 0.2 * sin(2*pi*step_freq*t_accel) + 0.05 * randn(size(t_accel));

% Y-axis: side-to-side motion
accel_y = 0.15 * sin(2*pi*step_freq*2*t_accel + pi/3) + 0.03 * randn(size(t_accel));

% Z-axis: vertical motion + gravity
accel_z = 9.81 + 0.3 * sin(2*pi*step_freq*4*t_accel) + 0.1 * randn(size(t_accel));

% Combine into single matrix
accelerometer_data = [t_accel, accel_x, accel_y, accel_z];
save('signals/raw/accelerometer.dat', 'accelerometer_data', '-ascii');

% Vibration data (mechanical system)
fs_vib = 1000;
duration_vib = 5;
t_vib = (0:1/fs_vib:duration_vib-1/fs_vib)';

% Simulate machinery vibration with multiple harmonics
fundamental_freq = 25; % 25 Hz fundamental frequency
vibration = 1.0 * sin(2*pi*fundamental_freq*t_vib) + ...
            0.5 * sin(2*pi*2*fundamental_freq*t_vib) + ...
            0.3 * sin(2*pi*3*fundamental_freq*t_vib) + ...
            0.2 * sin(2*pi*4*fundamental_freq*t_vib);

% Add transient events (impacts)
for impact_time = [1.2, 2.8, 4.1]
    impact_mask = abs(t_vib - impact_time) < 0.01;
    vibration(impact_mask) = vibration(impact_mask) + ...
        5 * exp(-abs(t_vib(impact_mask) - impact_time)/0.005);
end

% Add background noise
vibration = vibration + 0.1 * randn(size(vibration));

save('signals/raw/vibration_data.dat', 'vibration', '-ascii');

% Radar echo simulation
fs_radar = 1000;
duration_radar = 0.1; % 100 ms pulse
t_radar = (0:1/fs_radar:duration_radar-1/fs_radar)';

% Transmitted pulse
pulse_width = 0.001; % 1 ms pulse
pulse_signal = rectpuls((t_radar - 0.01)/pulse_width);

% Echo from multiple targets at different ranges
echo_signal = zeros(size(t_radar));

% Target distances (converted to time delays)
target_ranges = [100, 250, 400]; % meters
speed_of_light = 3e8; % m/s
target_delays = 2 * target_ranges / speed_of_light; % Two-way travel time

target_strengths = [1.0, 0.6, 0.3]; % Different reflectivity

for i = 1:length(target_delays)
    delayed_pulse = rectpuls((t_radar - target_delays(i) - 0.01)/pulse_width);
    echo_signal = echo_signal + target_strengths(i) * delayed_pulse;
end

% Add noise
radar_noise = 0.1 * randn(size(echo_signal));
radar_echo = pulse_signal + echo_signal + radar_noise;

save('signals/raw/radar_echo.dat', 'radar_echo', '-ascii');

%% Synthetic Signals
fprintf('Creating synthetic signals...\n');

% Common parameters for synthetic signals
fs_synth = 1000;
duration_synth = 2;
t_synth = (0:1/fs_synth:duration_synth-1/fs_synth)';

% AM modulated signal
carrier_freq = 100;
modulation_freq = 10;
modulation_depth = 0.8;

am_signal = (1 + modulation_depth * sin(2*pi*modulation_freq*t_synth)) .* ...
            sin(2*pi*carrier_freq*t_synth);

save('signals/synthetic/am_modulated.dat', 'am_signal', '-ascii');

% FM modulated signal
freq_deviation = 20;
fm_signal = sin(2*pi*carrier_freq*t_synth + ...
               (freq_deviation/modulation_freq)*sin(2*pi*modulation_freq*t_synth));

save('signals/synthetic/fm_modulated.dat', 'fm_signal', '-ascii');

% Pulse train
pulse_freq = 50;
duty_cycle = 0.2;
pulse_train = pulstran(t_synth, (0:1/pulse_freq:duration_synth)', 'rectpuls', duty_cycle/pulse_freq);

save('signals/synthetic/pulse_train.dat', 'pulse_train', '-ascii');

% White noise
white_noise = randn(size(t_synth));
save('signals/synthetic/white_noise.dat', 'white_noise', '-ascii');

% Swept sine (chirp) for testing
swept_sine = chirp(t_synth, 10, duration_synth, 200, 'logarithmic');
save('signals/synthetic/swept_sine.dat', 'swept_sine', '-ascii');

% Multi-tone signal for filter testing
freq1 = 50;
freq2 = 150;
freq3 = 300;

multitone = sin(2*pi*freq1*t_synth) + ...
            0.7*sin(2*pi*freq2*t_synth) + ...
            0.4*sin(2*pi*freq3*t_synth);

save('signals/synthetic/multitone.dat', 'multitone', '-ascii');

fprintf('Successfully created all signal files!\n');
fprintf('Generated:\n');
fprintf('  Audio files (WAV format):\n');
fprintf('    - Pure 440 Hz sine wave\n');
fprintf('    - Frequency chirp signal\n');
fprintf('    - Musical chord sample\n');
fprintf('    - Speech with background noise\n');
fprintf('  Raw signal data (ASCII format):\n');
fprintf('    - ECG signal simulation\n');
fprintf('    - 3-axis accelerometer data\n');
fprintf('    - Mechanical vibration data\n');
fprintf('    - Radar echo simulation\n');
fprintf('  Synthetic signals (ASCII format):\n');
fprintf('    - AM and FM modulated signals\n');
fprintf('    - Pulse train and white noise\n');
fprintf('    - Swept sine and multitone test signals\n');
fprintf('\nAll signal files are ready for signal processing exercises.\n');