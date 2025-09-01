% File location: OctaveMasterPro/utils/data_loader.m
% Comprehensive data loading utilities for OctaveMasterPro

function data_loader()
    % Data loading utility collection for OctaveMasterPro
    % Usage: Functions can be called individually or access help with data_loader_help()
end

function data_loader_help()
    % Display available data loading utilities
    fprintf('\n=== OctaveMasterPro Data Loader Utilities ===\n\n');
    fprintf('📁 CSV LOADERS:\n');
    fprintf('  load_employee_data()              - Load and validate employee dataset\n');
    fprintf('  load_sales_data(date_range)       - Load sales data with date filtering\n');
    fprintf('  load_stock_data(symbols)          - Load stock data for specific symbols\n');
    fprintf('  load_sensor_data(sensor_types)    - Load IoT sensor readings\n');
    fprintf('  load_experiment_data(groups)      - Load scientific experiment data\n\n');
    
    fprintf('🔬 MAT FILE LOADERS:\n');
    fprintf('  load_sensor_matrices()            - Load multi-dimensional sensor data\n');
    fprintf('  load_signal_analysis_data()       - Load signal processing datasets\n');
    fprintf('  load_optimization_data()          - Load optimization problem data\n\n');
    
    fprintf('🖼️ IMAGE LOADERS:\n');
    fprintf('  load_sample_images(image_type)    - Load image processing samples\n');
    fprintf('  load_batch_images()               - Load images for batch processing\n');
    fprintf('  load_medical_images()             - Load medical imaging examples\n\n');
    
    fprintf('🎵 SIGNAL LOADERS:\n');
    fprintf('  load_audio_signals(signal_type)   - Load audio files\n');
    fprintf('  load_raw_signals(data_type)       - Load raw sensor signals\n');
    fprintf('  load_synthetic_signals()          - Load generated test signals\n\n');
    
    fprintf('⚙️ UTILITIES:\n');
    fprintf('  validate_dataset(data, schema)    - Validate data integrity\n');
    fprintf('  get_dataset_info(dataset_name)    - Get dataset metadata\n');
    fprintf('  cache_dataset(data, cache_name)   - Cache processed data\n');
    fprintf('  list_available_datasets()         - Show all available datasets\n\n');
end

function data = load_employee_data(varargin)
    % Load and validate employee dataset with optional filtering
    % Usage: data = load_employee_data('Department', 'Engineering', 'Active', true)
    
    try
        data = readtable('datasets/data1.csv');
    catch
        error('Cannot load datasets/data1.csv - check file path and permissions');
    end
    
    % Parse optional filters
    p = inputParser;
    addParameter(p, 'Department', '');
    addParameter(p, 'Active', []);
    addParameter(p, 'MinAge', 0);
    addParameter(p, 'MaxAge', 100);
    addParameter(p, 'MinSalary', 0);
    parse(p, varargin{:});
    
    % Apply filters
    if ~isempty(p.Results.Department)
        dept_mask = strcmp(data.Department, p.Results.Department);
        data = data(dept_mask, :);
    end
    
    if ~isempty(p.Results.Active)
        if p.Results.Active
            active_mask = strcmp(data.Active, 'Yes');
        else
            active_mask = strcmp(data.Active, 'No');
        end
        data = data(active_mask, :);
    end
    
    if p.Results.MinAge > 0 || p.Results.MaxAge < 100
        age_mask = (data.Age >= p.Results.MinAge) & (data.Age <= p.Results.MaxAge);
        data = data(age_mask, :);
    end
    
    if p.Results.MinSalary > 0
        salary_mask = data.Salary >= p.Results.MinSalary;
        data = data(salary_mask, :);
    end
    
    fprintf('Loaded employee data: %d records\n', height(data));
    validate_employee_data(data);
end

function data = load_sales_data(varargin)
    % Load sales data with date range and filtering options
    % Usage: data = load_sales_data('StartDate', '2022-01-01', 'Region', 'North')
    
    try
        data = readtable('datasets/sales_data.csv');
    catch
        error('Cannot load datasets/sales_data.csv - check file path');
    end
    
    % Parse filters
    p = inputParser;
    addParameter(p, 'StartDate', '2022-01-01');
    addParameter(p, 'EndDate', '2023-12-31');
    addParameter(p, 'Region', '');
    addParameter(p, 'Product', '');
    parse(p, varargin{:});
    
    % Apply date filter
    start_date = datetime(p.Results.StartDate);
    end_date = datetime(p.Results.EndDate);
    data_dates = datetime(data.Date);
    date_mask = (data_dates >= start_date) & (data_dates <= end_date);
    data = data(date_mask, :);
    
    % Apply region filter
    if ~isempty(p.Results.Region)
        region_mask = strcmp(data.Region, p.Results.Region);
        data = data(region_mask, :);
    end
    
    % Apply product filter
    if ~isempty(p.Results.Product)
        product_mask = strcmp(data.Product, p.Results.Product);
        data = data(product_mask, :);
    end
    
    fprintf('Loaded sales data: %d records\n', height(data));
end

function data = load_stock_data(symbols)
    % Load stock data for specific symbols
    % Usage: data = load_stock_data({'AAPL', 'MSFT'}) or data = load_stock_data('AAPL')
    
    try
        all_data = readtable('datasets/stock_prices.csv');
    catch
        error('Cannot load datasets/stock_prices.csv - check file path');
    end
    
    % Convert single symbol to cell array
    if ischar(symbols)
        symbols = {symbols};
    end
    
    % Filter for requested symbols
    symbol_mask = false(height(all_data), 1);
    for i = 1:length(symbols)
        symbol_mask = symbol_mask | strcmp(all_data.Symbol, symbols{i});
    end
    
    data = all_data(symbol_mask, :);
    fprintf('Loaded stock data: %d records for symbols: %s\n', height(data), strjoin(symbols, ', '));
    validate_stock_data(data);
end

function signals = load_raw_signals(data_type)
    % Load raw signal data files
    % Usage: signals = load_raw_signals('ecg') % 'ecg', 'accelerometer', 'vibration', 'radar'
    
    base_dir = 'datasets/signals/raw/';
    
    switch lower(data_type)
        case 'ecg'
            filename = 'ecg_signal.dat';
        case 'accelerometer' 
            filename = 'accelerometer.dat';
        case 'vibration'
            filename = 'vibration_data.dat';
        case 'radar'
            filename = 'radar_echo.dat';
        otherwise
            error('Unknown data type: %s', data_type);
    end
    
    full_path = [base_dir filename];
    try
        signals = load(full_path);
        fprintf('Loaded raw signal: %s (%d samples)\n', filename, length(signals));
    catch
        error('Cannot load signal file: %s', full_path);
    end
end

function validate_employee_data(data)
    % Specific validation for employee dataset
    required_cols = {'ID', 'Name', 'Age', 'Salary', 'Department', 'Performance_Score'};
    
    % Check columns exist
    missing_cols = setdiff(required_cols, data.Properties.VariableNames);
    if ~isempty(missing_cols)
        error('Missing columns: %s', strjoin(missing_cols, ', '));
    end
    
    % Business rule validation
    if any(data.Age < 18 | data.Age > 80)
        warning('Some ages outside expected range (18-80)');
    end
    
    if any(data.Salary < 20000 | data.Salary > 200000)
        warning('Some salaries outside expected range ($20K-$200K)');
    end
    
    fprintf('Employee data validation complete\n');
end

function validate_stock_data(data)
    % Specific validation for stock price data
    invalid_high = data.High < max(data.Open, data.Close);
    invalid_low = data.Low > min(data.Open, data.Close);
    
    if any(invalid_high)
        warning('%d rows have High < max(Open, Close)', sum(invalid_high));
    end
    
    if any(invalid_low)
        warning('%d rows have Low > min(Open, Close)', sum(invalid_low));
    end
    
    fprintf('Stock data validation complete\n');
end

function sensor_data = load_sensor_matrices()
    % Load multi-dimensional sensor data from MAT file
    % Usage: sensor_data = load_sensor_matrices()
    
    try
        loaded = load('datasets/sensor_data.mat');
    catch
        error('Cannot load datasets/sensor_data.mat - run create_sensor_data.m first');
    end
    
    % Package into structure for easy access
    sensor_data.temperature_matrix = loaded.temperature_matrix;
    sensor_data.pressure_readings = loaded.pressure_readings;
    sensor_data.humidity_data = loaded.humidity_data;
    sensor_data.vibration_x = loaded.vibration_x;
    sensor_data.vibration_y = loaded.vibration_y;
    sensor_data.vibration_z = loaded.vibration_z;
    sensor_data.ph_readings = loaded.ph_readings;
    sensor_data.timestamps = loaded.timestamps;
    sensor_data.metadata = loaded.metadata;
    
    fprintf('Loaded sensor matrices: %dx%dx%d temperature data\n', size(sensor_data.temperature_matrix));
end

function signal_data = load_signal_analysis_data()
    % Load comprehensive signal analysis dataset
    % Usage: signal_data = load_signal_analysis_data()
    
    try
        loaded = load('datasets/signal_analysis.mat');
    catch
        error('Cannot load datasets/signal_analysis.mat - run create_signal_analysis.m first');
    end
    
    signal_data = loaded;
    fprintf('Loaded signal analysis data: %d samples at %d Hz\n', length(signal_data.clean_signal), signal_data.fs);
end

function opt_data = load_optimization_data()
    % Load optimization problem data
    % Usage: opt_data = load_optimization_data()
    
    try
        loaded = load('datasets/optimization_data.mat');
        opt_data = loaded;
        fprintf('Loaded optimization data\n');
    catch
        error('Cannot load datasets/optimization_data.mat - create file first');
    end
end

function data = load_sensor_data(varargin)
    % Load IoT sensor readings with filtering
    % Usage: data = load_sensor_data('SensorType', 'TEMP', 'Status', 'Normal')
    
    try
        data = readtable('datasets/sensor_readings.csv');
    catch
        error('Cannot load datasets/sensor_readings.csv - check file path');
    end
    
    % Parse filters
    p = inputParser;
    addParameter(p, 'SensorType', '');
    addParameter(p, 'Status', '');
    parse(p, varargin{:});
    
    % Apply filters
    if ~isempty(p.Results.SensorType)
        sensor_mask = contains(data.Sensor_ID, p.Results.SensorType);
        data = data(sensor_mask, :);
    end
    
    if ~isempty(p.Results.Status)
        status_mask = strcmp(data.Status, p.Results.Status);
        data = data(status_mask, :);
    end
    
    fprintf('Loaded sensor data: %d records\n', height(data));
end

function data = load_experiment_data(groups)
    % Load scientific experiment data with group filtering
    % Usage: data = load_experiment_data({'Control', 'Treatment_A'})
    
    try
        data = readtable('datasets/experiment_data.csv');
    catch
        error('Cannot load datasets/experiment_data.csv - check file path');
    end
    
    % Filter by groups if specified
    if exist('groups', 'var') && ~isempty(groups)
        if ischar(groups)
            groups = {groups};
        end
        
        group_mask = false(height(data), 1);
        for i = 1:length(groups)
            group_mask = group_mask | strcmp(data.Group, groups{i});
        end
        data = data(group_mask, :);
    end
    
    % Add derived variables
    data.Improvement = data.Post_Test - data.Pre_Test;
    fprintf('Loaded experiment data: %d records\n', height(data));
end

function images = load_sample_images(image_type)
    % Load sample images for processing
    % Usage: images = load_sample_images('samples') % 'samples', 'batch', 'medical', 'test'
    
    if nargin < 1
        image_type = 'samples';
    end
    
    switch lower(image_type)
        case 'samples'
            base_dir = 'datasets/images/samples/';
            pattern = 'sample_*.jpg';
        case 'batch'
            base_dir = 'datasets/images/batch/';
            pattern = 'batch_*.jpg';
        case 'medical'
            base_dir = 'datasets/images/medical/';
            pattern = '*.*';
        case 'test'
            base_dir = 'datasets/images/test/';
            pattern = '*.*';
        otherwise
            error('Unknown image type: %s', image_type);
    end
    
    file_list = dir([base_dir pattern]);
    images = cell(length(file_list), 1);
    
    for i = 1:length(file_list)
        images{i} = imread([base_dir file_list(i).name]);
    end
    
    fprintf('Loaded %d images from %s\n', length(file_list), base_dir);
end

function images = load_batch_images()
    % Load images for batch processing
    % Usage: images = load_batch_images()
    images = load_sample_images('batch');
end

function images = load_medical_images()
    % Load medical imaging examples
    % Usage: images = load_medical_images()
    images = load_sample_images('medical');
end

function [audio_data, fs] = load_audio_signals(signal_type)
    % Load audio signals
    % Usage: [audio, fs] = load_audio_signals('sine')
    
    base_dir = 'datasets/signals/audio/';
    
    switch lower(signal_type)
        case 'sine'
            filename = 'sine_wave_440hz.wav';
        case 'chirp'
            filename = 'chirp_signal.wav';
        case 'music'
            filename = 'music_sample.wav';
        case 'speech'
            filename = 'noisy_speech.wav';
        otherwise
            error('Unknown signal type: %s', signal_type);
    end
    
    [audio_data, fs] = audioread([base_dir filename]);
    fprintf('Loaded audio: %s\n', filename);
end

function signals = load_synthetic_signals()
    % Load generated test signals
    % Usage: signals = load_synthetic_signals()
    
    base_dir = 'datasets/signals/synthetic/';
    file_list = dir([base_dir '*.dat']);
    
    signals = struct();
    for i = 1:length(file_list)
        field_name = strrep(file_list(i).name, '.dat', '');
        signals.(field_name) = load([base_dir file_list(i).name]);
    end
    
    fprintf('Loaded %d synthetic signals\n', length(fieldnames(signals)));
end

function validate_dataset(data, expected_columns)
    % Validate dataset structure and content
    % Usage: validate_dataset(data_table, {'Column1', 'Column2'})
    
    if isempty(data)
        error('Dataset is empty');
    end
    
    fprintf('Dataset validation: %d rows x %d columns\n', size(data, 1), size(data, 2));
    
    if exist('expected_columns', 'var') && ~isempty(expected_columns)
        if istable(data)
            data_columns = data.Properties.VariableNames;
            missing_cols = setdiff(expected_columns, data_columns);
            if ~isempty(missing_cols)
                error('Missing columns: %s', strjoin(missing_cols, ', '));
            end
        end
    end
end

function dataset_info = get_dataset_info(dataset_name)
    % Get metadata for datasets
    % Usage: info = get_dataset_info('employee')
    
    switch lower(dataset_name)
        case {'employee', 'data1'}
            dataset_info.name = 'Employee Dataset';
            dataset_info.filename = 'data1.csv';
            dataset_info.rows = 1000;
            dataset_info.columns = 8;
            
        case 'sales'
            dataset_info.name = 'Sales Analytics Dataset';
            dataset_info.filename = 'sales_data.csv';
            dataset_info.rows = 2400;
            dataset_info.columns = 7;
            
        otherwise
            error('Unknown dataset: %s', dataset_name);
    end
    
    fprintf('%s: %d rows x %d columns\n', dataset_info.name, dataset_info.rows, dataset_info.columns);
end

function list_available_datasets()
    % List all available datasets
    % Usage: list_available_datasets()
    
    fprintf('\nAvailable datasets:\n');
    fprintf('  CSV: data1.csv, sales_data.csv, stock_prices.csv, sensor_readings.csv, experiment_data.csv\n');
    fprintf('  MAT: sensor_data.mat, signal_analysis.mat\n');
    fprintf('  Images: samples/, batch/, medical/, test/\n');
    fprintf('  Signals: audio/, raw/, synthetic/\n');
end

function cached_data = cache_dataset(data, cache_name)
    % Cache processed data
    % Usage: cache_dataset(processed_data, 'analysis_cache')
    
    cache_dir = 'cache/';
    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
    
    save([cache_dir cache_name '.mat'], 'data', '-v7');
    fprintf('Data cached to: %s\n', [cache_name '.mat']);
    cached_data = data;
end