% File location: OctaveMasterPro/flagship_project/project_scripts/data_ingestion.m
% Data ingestion module for IoT predictive maintenance project

function all_data = data_ingestion()
    % Main data ingestion pipeline
    % Returns: Structure containing all loaded datasets
    
    fprintf('Starting data ingestion pipeline...\n');
    
    % Initialize data structure
    all_data = struct();
    
    % Load core sensor readings
    fprintf('Loading sensor readings...\n');
    try
        all_data.sensors = load_sensor_data();
        fprintf('  SUCCESS: %d sensor readings loaded\n', height(all_data.sensors));
    catch ME
        fprintf('  ERROR loading sensor data: %s\n', ME.message);
        all_data.sensors = [];
    end
    
    % Load equipment metadata
    fprintf('Loading equipment metadata...\n');
    try
        all_data.equipment = readtable('../datasets/equipment_metadata.csv');
        fprintf('  SUCCESS: %d equipment records loaded\n', height(all_data.equipment));
    catch ME
        fprintf('  WARNING: Using generated equipment data\n');
        all_data.equipment = generate_equipment_metadata();
    end
    
    % Load maintenance logs
    fprintf('Loading maintenance logs...\n');
    try
        all_data.maintenance = readtable('../datasets/maintenance_logs.csv');
        fprintf('  SUCCESS: %d maintenance records loaded\n', height(all_data.maintenance));
    catch ME
        fprintf('  WARNING: Using generated maintenance data\n');
        all_data.maintenance = generate_maintenance_logs();
    end
    
    % Load failure events
    fprintf('Loading failure events...\n');
    try
        all_data.failures = readtable('../datasets/failure_events.csv');
        fprintf('  SUCCESS: %d failure events loaded\n', height(all_data.failures));
    catch ME
        fprintf('  WARNING: Using generated failure data\n');
        all_data.failures = generate_failure_events();
    end
    
    % Data validation
    validate_data_integrity(all_data);
    
    % Generate summary statistics
    all_data.summary = generate_data_summary(all_data);
    
    fprintf('Data ingestion completed successfully\n');
end

function equipment_data = generate_equipment_metadata()
    % Generate synthetic equipment metadata
    
    equipment_ids = {'EQ_001', 'EQ_002', 'EQ_003', 'EQ_004', 'EQ_005', ...
                    'EQ_006', 'EQ_007', 'EQ_008', 'EQ_009', 'EQ_010'};
    equipment_types = {'Pump', 'Motor', 'Compressor', 'Generator', 'Turbine'};
    locations = {'Building_A', 'Building_B', 'Building_C', 'Outdoor_North', 'Outdoor_South'};
    
    n_equipment = length(equipment_ids);
    
    % Create equipment table
    equipment_data = table();
    equipment_data.Equipment_ID = equipment_ids';
    equipment_data.Equipment_Type = equipment_types(randi(length(equipment_types), n_equipment, 1))';
    equipment_data.Location = locations(randi(length(locations), n_equipment, 1))';
    equipment_data.Install_Date = datetime('2020-01-01') + days(randi(365*3, n_equipment, 1));
    equipment_data.Rated_Power = 50 + randi(200, n_equipment, 1); % kW
    equipment_data.Operating_Hours = 1000 + randi(8000, n_equipment, 1);
    
    % Map sensors to equipment
    sensor_mapping = {
        'EQ_001', 'TEMP_001'; 'EQ_001', 'VIB_001';
        'EQ_002', 'TEMP_002'; 'EQ_002', 'PRES_001';
        'EQ_003', 'TEMP_003'; 'EQ_003', 'VIB_002';
        'EQ_004', 'HUM_001'; 'EQ_004', 'MULTI_001';
        'EQ_005', 'HUM_002'; 'EQ_005', 'MULTI_002';
        'EQ_006', 'PRES_002'; 'EQ_006', 'MULTI_003';
        'EQ_007', 'TEMP_001'; 'EQ_008', 'TEMP_002';
        'EQ_009', 'VIB_001'; 'EQ_010', 'VIB_002'
    };
    
    % Add sensor mapping to equipment data
    equipment_data.Primary_Sensor = cell(n_equipment, 1);
    for i = 1:n_equipment
        eq_id = equipment_data.Equipment_ID{i};
        sensor_idx = find(strcmp(sensor_mapping(:, 1), eq_id), 1);
        if ~isempty(sensor_idx)
            equipment_data.Primary_Sensor{i} = sensor_mapping{sensor_idx, 2};
        else
            equipment_data.Primary_Sensor{i} = 'MULTI_001'; % Default
        end
    end
    
    fprintf('Generated %d equipment records\n', height(equipment_data));
end

function maintenance_data = generate_maintenance_logs()
    % Generate synthetic maintenance history
    
    equipment_ids = {'EQ_001', 'EQ_002', 'EQ_003', 'EQ_004', 'EQ_005', ...
                    'EQ_006', 'EQ_007', 'EQ_008', 'EQ_009', 'EQ_010'};
    maintenance_types = {'Routine', 'Preventive', 'Corrective', 'Emergency'};
    
    % Generate 200 maintenance events over 2 years
    n_events = 200;
    
    maintenance_data = table();
    maintenance_data.Maintenance_ID = arrayfun(@(x) sprintf('MAINT_%04d', x), 1:n_events, 'UniformOutput', false)';
    maintenance_data.Equipment_ID = equipment_ids(randi(length(equipment_ids), n_events, 1))';
    maintenance_data.Date = datetime('2022-01-01') + days(randi(730, n_events, 1));
    maintenance_data.Type = maintenance_types(randi(length(maintenance_types), n_events, 1))';
    
    % Cost varies by maintenance type
    base_costs = containers.Map(maintenance_types, {500, 800, 1200, 2500});
    costs = zeros(n_events, 1);
    
    for i = 1:n_events
        base_cost = base_costs(maintenance_data.Type{i});
        costs(i) = base_cost + randn() * base_cost * 0.3; % ±30% variation
    end
    
    maintenance_data.Cost = max(100, costs); % Minimum $100
    maintenance_data.Duration_Hours = 2 + randi(20, n_events, 1); % 2-22 hours
    maintenance_data.Technician_ID = arrayfun(@(x) sprintf('TECH_%02d', randi(15)), 1:n_events, 'UniformOutput', false)';
    
    fprintf('Generated %d maintenance records\n', height(maintenance_data));
end

function failure_data = generate_failure_events()
    % Generate synthetic failure event history
    
    equipment_ids = {'EQ_001', 'EQ_002', 'EQ_003', 'EQ_004', 'EQ_005', ...
                    'EQ_006', 'EQ_007', 'EQ_008', 'EQ_009', 'EQ_010'};
    failure_modes = {'Overheating', 'Vibration_Excessive', 'Pressure_Loss', ...
                    'Electrical_Fault', 'Mechanical_Wear', 'Sensor_Malfunction'};
    
    % Generate 25 failure events (realistic failure rate)
    n_failures = 25;
    
    failure_data = table();
    failure_data.Failure_ID = arrayfun(@(x) sprintf('FAIL_%04d', x), 1:n_failures, 'UniformOutput', false)';
    failure_data.Equipment_ID = equipment_ids(randi(length(equipment_ids), n_failures, 1))';
    failure_data.Failure_Date = datetime('2022-01-01') + days(randi(700, n_failures, 1));
    failure_data.Failure_Mode = failure_modes(randi(length(failure_modes), n_failures, 1))';
    
    % Severity and costs
    failure_data.Severity = randi(5, n_failures, 1); % 1-5 scale
    failure_data.Repair_Cost = 1000 + randi(10000, n_failures, 1); % $1K-$11K
    failure_data.Downtime_Hours = 4 + randi(48, n_failures, 1); % 4-52 hours
    failure_data.Root_Cause = failure_modes(randi(length(failure_modes), n_failures, 1))';
    
    % Add detection method
    detection_methods = {'Sensor_Alert', 'Visual_Inspection', 'Performance_Drop', 'Scheduled_Check'};
    failure_data.Detection_Method = detection_methods(randi(length(detection_methods), n_failures, 1))';
    
    fprintf('Generated %d failure events\n', height(failure_data));
end

function validate_data_integrity(all_data)
    % Validate data integrity across all datasets
    
    fprintf('Validating data integrity...\n');
    
    % Check sensor data consistency
    if ~isempty(all_data.sensors)
        unique_sensors = unique(all_data.sensors.Sensor_ID);
        fprintf('  Sensors: %d unique sensor IDs\n', length(unique_sensors));
        
        % Check for data gaps
        time_gaps = diff(sort(unique(all_data.sensors.Timestamp)));
        large_gaps = time_gaps > hours(2);
        if any(large_gaps)
            fprintf('  WARNING: %d time gaps > 2 hours detected\n', sum(large_gaps));
        end
    end
    
    % Check equipment-sensor mapping
    if ~isempty(all_data.equipment) && ~isempty(all_data.sensors)
        equipment_sensors = all_data.equipment.Primary_Sensor;
        sensor_ids = unique(all_data.sensors.Sensor_ID);
        
        unmapped_sensors = setdiff(sensor_ids, equipment_sensors);
        if ~isempty(unmapped_sensors)
            fprintf('  WARNING: %d sensors not mapped to equipment\n', length(unmapped_sensors));
        end
    end
    
    % Check maintenance-failure correlation
    if ~isempty(all_data.maintenance) && ~isempty(all_data.failures)
        maint_equipment = unique(all_data.maintenance.Equipment_ID);
        failure_equipment = unique(all_data.failures.Equipment_ID);
        
        common_equipment = intersect(maint_equipment, failure_equipment);
        fprintf('  Equipment with both maintenance and failures: %d\n', length(common_equipment));
    end
    
    fprintf('Data validation completed\n');
end

function summary = generate_data_summary(all_data)
    % Generate comprehensive data summary
    
    summary = struct();
    
    % Sensor data summary
    if ~isempty(all_data.sensors)
        summary.sensor_count = length(unique(all_data.sensors.Sensor_ID));
        summary.date_range = [min(all_data.sensors.Timestamp), max(all_data.sensors.Timestamp)];
        summary.total_readings = height(all_data.sensors);
        summary.anomaly_rate = mean(strcmp(all_data.sensors.Status, 'Warning') | ...
                                   strcmp(all_data.sensors.Status, 'Critical'));
    end
    
    % Equipment summary
    if ~isempty(all_data.equipment)
        summary.equipment_count = height(all_data.equipment);
        summary.equipment_types = unique(all_data.equipment.Equipment_Type);
        summary.average_age_years = mean(years(datetime('now') - all_data.equipment.Install_Date));
    end
    
    % Maintenance summary
    if ~isempty(all_data.maintenance)
        summary.maintenance_events = height(all_data.maintenance);
        summary.total_maintenance_cost = sum(all_data.maintenance.Cost);
        summary.avg_maintenance_cost = mean(all_data.maintenance.Cost);
    end
    
    % Failure summary
    if ~isempty(all_data.failures)
        summary.failure_count = height(all_data.failures);
        summary.total_repair_cost = sum(all_data.failures.Repair_Cost);
        summary.avg_downtime_hours = mean(all_data.failures.Downtime_Hours);
        summary.failure_rate_percent = summary.failure_count / summary.equipment_count * 100;
    end
    
    % Display summary
    fprintf('\n=== DATA INGESTION SUMMARY ===\n');
    fprintf('Sensors: %d units, %d readings\n', summary.sensor_count, summary.total_readings);
    fprintf('Equipment: %d units, avg age %.1f years\n', summary.equipment_count, summary.average_age_years);
    fprintf('Maintenance: %d events, $%.0f total cost\n', summary.maintenance_events, summary.total_maintenance_cost);
    fprintf('Failures: %d events (%.1f%% rate), %.1f avg downtime hours\n', ...
            summary.failure_count, summary.failure_rate_percent, summary.avg_downtime_hours);
end