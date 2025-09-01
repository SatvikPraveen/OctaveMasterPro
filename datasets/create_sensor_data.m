% File location: OctaveMasterPro/datasets/create_sensor_data.m
% Script to generate sensor_data.mat for the OctaveMasterPro project

% Generate temperature matrix (100 sensors × 24 hours × 30 days)
fprintf('Generating temperature data...\n');
temperature_matrix = zeros(100, 24, 30);

for sensor = 1:100
  base_temp = 20 + randn() * 5; % Base temperature varies by sensor location
  
  for day = 1:30
    % Daily temperature cycle
    daily_pattern = 5 * sin(2*pi*(0:23)/24 - pi/2) + base_temp;
    
    % Add seasonal drift
    seasonal_drift = 2 * sin(2*pi*day/365);
    
    % Add random noise and sensor-specific bias
    sensor_bias = 0.5 * randn();
    noise = 0.3 * randn(1, 24);
    
    temperature_matrix(sensor, :, day) = daily_pattern + seasonal_drift + sensor_bias + noise;
  end
end

% Generate high-frequency pressure readings
fprintf('Generating pressure data...\n');
fs = 1000; % 1000 Hz sampling rate
t = (0:9999)/fs; % 10 seconds of data
base_pressure = 101325; % Standard atmospheric pressure in Pa

% Create realistic pressure signal with multiple components
pressure_readings = base_pressure + ...
                   50 * sin(2*pi*0.1*t) + ...      % Low frequency variation
                   20 * sin(2*pi*1*t) + ...        % 1 Hz component
                   10 * sin(2*pi*10*t) + ...       % 10 Hz component
                   5 * randn(size(t));             % Random noise

% Create metadata structure
fprintf('Creating metadata...\n');
metadata.sensor_count = 100;
metadata.sampling_rate = fs;
metadata.units.temperature = 'Celsius';
metadata.units.pressure = 'Pascal';
metadata.calibration_date = '2024-01-15';
metadata.location = 'Laboratory Building A';
metadata.experiment_id = 'EXP_2024_001';

% Create calibration data
metadata.calibration.temp_offset = randn(100, 1) * 0.1;
metadata.calibration.temp_gain = 1 + randn(100, 1) * 0.01;
metadata.calibration.pressure_offset = randn() * 10;
metadata.calibration.pressure_gain = 1 + randn() * 0.001;

% Generate timestamps for temperature data
fprintf('Generating timestamps...\n');
start_time = datenum('2024-01-01 00:00:00');
timestamps = [];

for day = 1:30
  for hour = 1:24
    timestamp = start_time + (day-1) + (hour-1)/24;
    timestamps(end+1) = timestamp;
  end
end

timestamps = reshape(timestamps, [24, 30]);

% Generate additional sensor arrays for variety
fprintf('Generating additional sensor data...\n');

% Humidity data (related to temperature)
humidity_data = 50 + 10 * sin(2*pi*(1:720)/24) + 5 * randn(1, 720);
humidity_data = max(10, min(90, humidity_data)); % Clamp to realistic range

% Vibration data (accelerometer-like)
vibration_x = 0.1 * randn(1, 5000);
vibration_y = 0.1 * randn(1, 5000);
vibration_z = 9.81 + 0.05 * randn(1, 5000); % Gravity + noise

% pH sensor data (for chemical processes)
ph_readings = 7 + 0.5 * sin(2*pi*(1:1440)/1440) + 0.1 * randn(1, 1440);
ph_readings = max(1, min(14, ph_readings)); % pH scale limits

% Save all data to MAT file
fprintf('Saving to sensor_data.mat...\n');
save('sensor_data.mat', 'temperature_matrix', 'pressure_readings', 'metadata', ...
     'timestamps', 'humidity_data', 'vibration_x', 'vibration_y', 'vibration_z', ...
     'ph_readings', '-v7');

fprintf('Successfully created sensor_data.mat\n');
fprintf('Contains:\n');
fprintf('  - temperature_matrix: %dx%dx%d (sensors x hours x days)\n', ...
        size(temperature_matrix, 1), size(temperature_matrix, 2), size(temperature_matrix, 3));
fprintf('  - pressure_readings: 1x%d high-frequency samples\n', length(pressure_readings));
fprintf('  - timestamps: %dx%d timestamp matrix\n', size(timestamps, 1), size(timestamps, 2));
fprintf('  - humidity_data: 1x%d hourly readings\n', length(humidity_data));
fprintf('  - vibration_x/y/z: 3-axis accelerometer data\n');
fprintf('  - ph_readings: 1x%d chemical sensor data\n', length(ph_readings));
fprintf('  - metadata: Comprehensive experiment information\n');