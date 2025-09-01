% File location: OctaveMasterPro/flagship_project/project_scripts/exploratory_analysis.m
% Exploratory data analysis module for IoT predictive maintenance project

function analysis_results = exploratory_analysis(processed_data)
    % Comprehensive exploratory data analysis
    % Input: processed_data structure from data_preprocessing
    % Returns: analysis results and generates visualizations
    
    fprintf('Starting exploratory data analysis...\n');
    
    % Initialize results structure
    analysis_results = struct();
    
    % Basic data overview
    analysis_results.data_summary = analyze_data_overview(processed_data);
    
    % Sensor data analysis
    analysis_results.sensor_analysis = analyze_sensor_patterns(processed_data.sensors);
    
    % Equipment analysis
    analysis_results.equipment_analysis = analyze_equipment_characteristics(processed_data.equipment);
    
    % Maintenance analysis
    analysis_results.maintenance_analysis = analyze_maintenance_patterns(processed_data.maintenance);
    
    % Failure analysis
    analysis_results.failure_analysis = analyze_failure_patterns(processed_data.failures);
    
    % Correlation analysis
    analysis_results.correlations = analyze_correlations(processed_data.master_dataset);
    
    % Feature importance analysis
    analysis_results.feature_importance = analyze_feature_importance(processed_data.master_dataset);
    
    % Generate comprehensive visualizations
    generate_analysis_plots(processed_data, analysis_results);
    
    fprintf('Exploratory analysis completed\n');
end

function data_summary = analyze_data_overview(processed_data)
    % Generate basic data overview statistics
    
    fprintf('Analyzing data overview...\n');
    
    data_summary = struct();
    
    % Data volumes
    data_summary.sensor_records = height(processed_data.sensors);
    data_summary.equipment_count = height(processed_data.equipment);
    data_summary.maintenance_records = height(processed_data.maintenance);
    data_summary.failure_events = height(processed_data.failures);
    data_summary.master_dataset_size = height(processed_data.master_dataset);
    
    % Time range
    data_summary.date_range.start = min(processed_data.sensors.Timestamp);
    data_summary.date_range.end = max(processed_data.sensors.Timestamp);
    data_summary.analysis_period_days = days(data_summary.date_range.end - data_summary.date_range.start);
    
    % Data quality metrics
    sensor_data = processed_data.sensors;
    data_summary.data_quality.sensor_completeness = ...
        (sum(~isnan(sensor_data.Temperature)) + sum(~isnan(sensor_data.Pressure)) + ...
         sum(~isnan(sensor_data.Humidity)) + sum(~isnan(sensor_data.Vibration))) / ...
        (4 * height(sensor_data)) * 100;
    
    % Equipment distribution
    equipment_types = categories(processed_data.equipment.Equipment_Type);
    for i = 1:length(equipment_types)
        type_count = sum(processed_data.equipment.Equipment_Type == equipment_types{i});
        data_summary.equipment_distribution.(matlab.lang.makeValidName(char(equipment_types{i}))) = type_count;
    end
    
    fprintf('Data overview analysis completed\n');
end

function sensor_analysis = analyze_sensor_patterns(sensors)
    % Analyze sensor data patterns and trends
    
    fprintf('Analyzing sensor patterns...\n');
    
    sensor_analysis = struct();
    
    % Basic statistics
    numeric_columns = {'Temperature', 'Pressure', 'Humidity', 'Vibration'};
    
    for i = 1:length(numeric_columns)
        col_name = numeric_columns{i};
        col_data = sensors.(col_name);
        
        stats = struct();
        stats.mean = mean(col_data, 'omitnan');
        stats.std = std(col_data, 'omitnan');
        stats.min = min(col_data);
        stats.max = max(col_data);
        stats.median = median(col_data, 'omitnan');
        stats.q25 = prctile(col_data, 25);
        stats.q75 = prctile(col_data, 75);
        stats.skewness = skewness(col_data);
        
        sensor_analysis.statistics.(col_name) = stats;
    end
    
    % Temporal patterns
    sensors_sorted = sortrows(sensors, 'Timestamp');
    hourly_avg = struct();
    
    for i = 1:length(numeric_columns)
        col_name = numeric_columns{i};
        
        % Group by hour of day
        hours = hour(sensors_sorted.Timestamp);
        hourly_values = accumarray(hours + 1, sensors_sorted.(col_name), [24, 1], @mean);
        hourly_avg.(col_name) = hourly_values;
    end
    
    sensor_analysis.temporal_patterns.hourly = hourly_avg;
    
    % Sensor-specific analysis
    unique_sensors = unique(sensors.Sensor_ID);
    sensor_analysis.sensor_count = length(unique_sensors);
    
    sensor_reliability = struct();
    for i = 1:min(10, length(unique_sensors)) % Top 10 sensors
        sensor_id = unique_sensors{i};
        sensor_mask = strcmp(sensors.Sensor_ID, sensor_id);
        sensor_data = sensors(sensor_mask, :);
        
        reliability = struct();
        reliability.data_points = height(sensor_data);
        reliability.completeness = sum(~isnan(sensor_data.Temperature)) / height(sensor_data) * 100;
        
        % Calculate coefficient of variation for stability
        reliability.temperature_stability = std(sensor_data.Temperature, 'omitnan') / ...
                                           mean(sensor_data.Temperature, 'omitnan');
        
        sensor_reliability.(matlab.lang.makeValidName(sensor_id)) = reliability;
    end
    
    sensor_analysis.sensor_reliability = sensor_reliability;
    
    fprintf('Sensor pattern analysis completed\n');
end

function equipment_analysis = analyze_equipment_characteristics(equipment)
    % Analyze equipment characteristics and distributions
    
    fprintf('Analyzing equipment characteristics...\n');
    
    equipment_analysis = struct();
    
    % Age distribution
    equipment_analysis.age_stats.mean_age_days = mean(equipment.Age_Days);
    equipment_analysis.age_stats.median_age_days = median(equipment.Age_Days);
    equipment_analysis.age_stats.min_age_days = min(equipment.Age_Days);
    equipment_analysis.age_stats.max_age_days = max(equipment.Age_Days);
    
    % Operational hours analysis
    equipment_analysis.usage_stats.mean_hours = mean(equipment.Operational_Hours);
    equipment_analysis.usage_stats.median_hours = median(equipment.Operational_Hours);
    
    % Equipment type analysis
    type_counts = countcats(equipment.Equipment_Type);
    type_names = categories(equipment.Equipment_Type);
    
    for i = 1:length(type_names)
        type_data = equipment(equipment.Equipment_Type == type_names{i}, :);
        
        type_stats = struct();
        type_stats.count = type_counts(i);
        type_stats.avg_age = mean(type_data.Age_Days);
        type_stats.avg_operational_hours = mean(type_data.Operational_Hours);
        type_stats.avg_failure_rate = mean(type_data.Historical_Failure_Rate);
        type_stats.avg_maintenance_efficiency = mean(type_data.Maintenance_Efficiency);
        
        equipment_analysis.by_type.(matlab.lang.makeValidName(char(type_names{i}))) = type_stats;
    end
    
    % Location analysis
    location_counts = countcats(equipment.Location);
    location_names = categories(equipment.Location);
    
    for i = 1:length(location_names)
        location_data = equipment(equipment.Location == location_names{i}, :);
        
        location_stats = struct();
        location_stats.count = location_counts(i);
        location_stats.avg_failure_rate = mean(location_data.Historical_Failure_Rate);
        
        equipment_analysis.by_location.(matlab.lang.makeValidName(char(location_names{i}))) = location_stats;
    end
    
    fprintf('Equipment analysis completed\n');
end

function maintenance_analysis = analyze_maintenance_patterns(maintenance)
    % Analyze maintenance patterns and effectiveness
    
    fprintf('Analyzing maintenance patterns...\n');
    
    maintenance_analysis = struct();
    
    % Maintenance frequency
    maintenance_analysis.total_events = height(maintenance);
    
    % By type
    type_counts = countcats(maintenance.Maintenance_Type);
    type_names = categories(maintenance.Maintenance_Type);
    
    for i = 1:length(type_names)
        type_data = maintenance(maintenance.Maintenance_Type == type_names{i}, :);
        
        type_stats = struct();
        type_stats.count = type_counts(i);
        type_stats.percentage = type_counts(i) / height(maintenance) * 100;
        type_stats.avg_cost = mean(type_data.Cost, 'omitnan');
        type_stats.total_cost = sum(type_data.Cost, 'omitnan');
        
        if sum(~isnan(type_data.Days_Since_Last)) > 0
            type_stats.avg_interval = mean(type_data.Days_Since_Last, 'omitnan');
        else
            type_stats.avg_interval = NaN;
        end
        
        maintenance_analysis.by_type.(matlab.lang.makeValidName(char(type_names{i}))) = type_stats;
    end
    
    % Cost analysis
    maintenance_analysis.cost_stats.total_cost = sum(maintenance.Cost, 'omitnan');
    maintenance_analysis.cost_stats.avg_cost = mean(maintenance.Cost, 'omitnan');
    maintenance_analysis.cost_stats.median_cost = median(maintenance.Cost, 'omitnan');
    
    % Temporal patterns
    maintenance_sorted = sortrows(maintenance, 'Maintenance_Date');
    
    % Monthly maintenance volume
    months = month(maintenance_sorted.Maintenance_Date);
    monthly_counts = accumarray(months, ones(height(maintenance_sorted), 1), [12, 1], @sum);
    maintenance_analysis.temporal.monthly_distribution = monthly_counts;
    
    % Day of week patterns
    days_of_week = weekday(maintenance_sorted.Maintenance_Date);
    weekly_counts = accumarray(days_of_week, ones(height(maintenance_sorted), 1), [7, 1], @sum);
    maintenance_analysis.temporal.weekly_distribution = weekly_counts;
    
    fprintf('Maintenance pattern analysis completed\n');
end

function failure_analysis = analyze_failure_patterns(failures)
    % Analyze failure patterns and characteristics
    
    fprintf('Analyzing failure patterns...\n');
    
    failure_analysis = struct();
    
    % Basic statistics
    failure_analysis.total_failures = height(failures);
    
    % Failure type analysis
    type_counts = countcats(failures.Failure_Type);
    type_names = categories(failures.Failure_Type);
    
    for i = 1:length(type_names)
        type_data = failures(failures.Failure_Type == type_names{i}, :);
        
        type_stats = struct();
        type_stats.count = type_counts(i);
        type_stats.percentage = type_counts(i) / height(failures) * 100;
        type_stats.avg_downtime = mean(type_data.Downtime_Hours, 'omitnan');
        type_stats.total_downtime = sum(type_data.Downtime_Hours, 'omitnan');
        type_stats.avg_repair_cost = mean(type_data.Repair_Cost, 'omitnan');
        type_stats.total_repair_cost = sum(type_data.Repair_Cost, 'omitnan');
        
        failure_analysis.by_type.(matlab.lang.makeValidName(char(type_names{i}))) = type_stats;
    end
    
    % Severity analysis
    severity_counts = countcats(failures.Severity);
    severity_names = categories(failures.Severity);
    
    for i = 1:length(severity_names)
        severity_data = failures(failures.Severity == severity_names{i}, :);
        
        severity_stats = struct();
        severity_stats.count = severity_counts(i);
        severity_stats.percentage = severity_counts(i) / height(failures) * 100;
        severity_stats.avg_downtime = mean(severity_data.Downtime_Hours, 'omitnan');
        severity_stats.avg_repair_cost = mean(severity_data.Repair_Cost, 'omitnan');
        
        failure_analysis.by_severity.(matlab.lang.makeValidName(char(severity_names{i}))) = severity_stats;
    end
    
    % Time between failures analysis
    valid_tbf = failures.Time_Between_Failures(~isnan(failures.Time_Between_Failures));
    if ~isempty(valid_tbf)
        failure_analysis.time_between_failures.mean = mean(valid_tbf);
        failure_analysis.time_between_failures.median = median(valid_tbf);
        failure_analysis.time_between_failures.std = std(valid_tbf);
        failure_analysis.time_between_failures.min = min(valid_tbf);
        failure_analysis.time_between_failures.max = max(valid_tbf);
    end
    
    % Cost impact
    failure_analysis.cost_impact.total_repair_cost = sum(failures.Repair_Cost, 'omitnan');
    failure_analysis.cost_impact.avg_repair_cost = mean(failures.Repair_Cost, 'omitnan');
    failure_analysis.cost_impact.total_downtime_hours = sum(failures.Downtime_Hours, 'omitnan');
    failure_analysis.cost_impact.avg_downtime_hours = mean(failures.Downtime_Hours, 'omitnan');
    
    fprintf('Failure pattern analysis completed\n');
end

function correlations = analyze_correlations(master_dataset)
    % Analyze correlations between variables
    
    fprintf('Analyzing correlations...\n');
    
    correlations = struct();
    
    % Select numeric variables for correlation analysis
    numeric_vars = {'Avg_Temperature', 'Avg_Pressure', 'Avg_Humidity', 'Avg_Vibration', ...
                   'Max_Vibration', 'Std_Temperature', 'Sensor_Index', 'Age_Days', ...
                   'Operational_Hours', 'Historical_Failure_Rate', 'Maintenance_Efficiency', ...
                   'Days_Since_Maintenance'};
    
    % Create correlation matrix
    valid_vars = {};
    for i = 1:length(numeric_vars)
        if any(strcmp(master_dataset.Properties.VariableNames, numeric_vars{i}))
            valid_vars{end+1} = numeric_vars{i};
        end
    end
    
    if length(valid_vars) >= 2
        corr_data = table2array(master_dataset(:, valid_vars));
        
        % Remove rows with NaN values
        valid_rows = ~any(isnan(corr_data), 2);
        corr_data_clean = corr_data(valid_rows, :);
        
        if size(corr_data_clean, 1) > 1
            [R, P] = corrcoef(corr_data_clean);
            correlations.correlation_matrix = R;
            correlations.p_values = P;
            correlations.variable_names = valid_vars;
            
            % Find strongest correlations
            [max_corr, max_idx] = max(abs(R(triu(true(size(R)), 1))));
            [i, j] = find(abs(R) == max_corr & triu(true(size(R)), 1));
            
            if ~isempty(i)
                correlations.strongest_correlation.value = R(i(1), j(1));
                correlations.strongest_correlation.variables = {valid_vars{i(1)}, valid_vars{j(1)}};
            end
        end
    end
    
    % Correlation with failure targets
    if any(strcmp(master_dataset.Properties.VariableNames, 'Failure_Next_7_Days'))
        failure_7_corr = struct();
        failure_30_corr = struct();
        
        for i = 1:length(valid_vars)
            var_data = master_dataset.(valid_vars{i});
            valid_idx = ~isnan(var_data);
            
            if sum(valid_idx) > 10
                [r_7, p_7] = corrcoef(var_data(valid_idx), double(master_dataset.Failure_Next_7_Days(valid_idx)));
                [r_30, p_30] = corrcoef(var_data(valid_idx), double(master_dataset.Failure_Next_30_Days(valid_idx)));
                
                failure_7_corr.(valid_vars{i}) = struct('correlation', r_7(1,2), 'p_value', p_7(1,2));
                failure_30_corr.(valid_vars{i}) = struct('correlation', r_30(1,2), 'p_value', p_30(1,2));
            end
        end
        
        correlations.failure_correlations.next_7_days = failure_7_corr;
        correlations.failure_correlations.next_30_days = failure_30_corr;
    end
    
    fprintf('Correlation analysis completed\n');
end

function feature_importance = analyze_feature_importance(master_dataset)
    % Analyze feature importance using simple statistical methods
    
    fprintf('Analyzing feature importance...\n');
    
    feature_importance = struct();
    
    if ~any(strcmp(master_dataset.Properties.VariableNames, 'Failure_Next_7_Days'))
        fprintf('Warning: Target variable not found in dataset\n');
        return;
    end
    
    % Select features for importance analysis
    feature_vars = {'Avg_Temperature', 'Avg_Pressure', 'Avg_Humidity', 'Avg_Vibration', ...
                   'Max_Vibration', 'Std_Temperature', 'Sensor_Index', 'Age_Days', ...
                   'Operational_Hours', 'Historical_Failure_Rate', 'Maintenance_Efficiency', ...
                   'Days_Since_Maintenance'};
    
    valid_features = {};
    for i = 1:length(feature_vars)
        if any(strcmp(master_dataset.Properties.VariableNames, feature_vars{i}))
            valid_features{end+1} = feature_vars{i};
        end
    end
    
    % Statistical importance (using t-test)
    importance_7_days = struct();
    importance_30_days = struct();
    
    for i = 1:length(valid_features)
        feature_name = valid_features{i};
        feature_data = master_dataset.(feature_name);
        
        % Remove NaN values
        valid_idx = ~isnan(feature_data);
        
        if sum(valid_idx) > 10
            % 7-day prediction importance
            failure_7 = master_dataset.Failure_Next_7_Days(valid_idx);
            feature_clean = feature_data(valid_idx);
            
            % T-test for difference in means
            group1 = feature_clean(failure_7 == 1);
            group0 = feature_clean(failure_7 == 0);
            
            if length(group1) > 1 && length(group0) > 1
                [~, p_val, ~, stats] = ttest2(group1, group0);
                importance_7_days.(feature_name) = struct('t_stat', abs(stats.tstat), 'p_value', p_val, ...
                                                         'mean_failure', mean(group1), 'mean_normal', mean(group0));
            end
            
            % 30-day prediction importance
            if any(strcmp(master_dataset.Properties.VariableNames, 'Failure_Next_30_Days'))
                failure_30 = master_dataset.Failure_Next_30_Days(valid_idx);
                
                group1_30 = feature_clean(failure_30 == 1);
                group0_30 = feature_clean(failure_30 == 0);
                
                if length(group1_30) > 1 && length(group0_30) > 1
                    [~, p_val_30, ~, stats_30] = ttest2(group1_30, group0_30);
                    importance_30_days.(feature_name) = struct('t_stat', abs(stats_30.tstat), 'p_value', p_val_30, ...
                                                              'mean_failure', mean(group1_30), 'mean_normal', mean(group0_30));
                end
            end
        end
    end
    
    feature_importance.statistical_importance_7_days = importance_7_days;
    feature_importance.statistical_importance_30_days = importance_30_days;
    
    fprintf('Feature importance analysis completed\n');
end

function generate_analysis_plots(processed_data, analysis_results)
    % Generate comprehensive analysis visualizations
    
    fprintf('Generating analysis visualizations...\n');
    
    % Create figures directory
    if ~exist('report/figures', 'dir')
        mkdir('report/figures');
    end
    
    % 1. Sensor trends plot
    figure('Position', [100, 100, 1200, 800]);
    
    sensors = processed_data.sensors;
    sensors_sorted = sortrows(sensors, 'Timestamp');
    
    subplot(2, 2, 1);
    plot(sensors_sorted.Timestamp, sensors_sorted.Temperature);
    title('Temperature Trends');
    xlabel('Time');
    ylabel('Temperature (°C)');
    grid on;
    
    subplot(2, 2, 2);
    plot(sensors_sorted.Timestamp, sensors_sorted.Pressure);
    title('Pressure Trends');
    xlabel('Time');
    ylabel('Pressure (Pa)');
    grid on;
    
    subplot(2, 2, 3);
    plot(sensors_sorted.Timestamp, sensors_sorted.Humidity);
    title('Humidity Trends');
    xlabel('Time');
    ylabel('Humidity (%)');
    grid on;
    
    subplot(2, 2, 4);
    plot(sensors_sorted.Timestamp, sensors_sorted.Vibration);
    title('Vibration Trends');
    xlabel('Time');
    ylabel('Vibration (g)');
    grid on;
    
    sgtitle('Sensor Data Trends Over Time');
    saveas(gcf, 'report/figures/sensor_trends.png');
    close;
    
    % 2. Equipment distribution plot
    figure('Position', [100, 100, 1000, 600]);
    
    equipment_types = categories(processed_data.equipment.Equipment_Type);
    type_counts = countcats(processed_data.equipment.Equipment_Type);
    
    subplot(1, 2, 1);
    pie(type_counts, equipment_types);
    title('Equipment Type Distribution');
    
    subplot(1, 2, 2);
    histogram(processed_data.equipment.Age_Days / 365);
    title('Equipment Age Distribution');
    xlabel('Age (Years)');
    ylabel('Count');
    grid on;
    
    saveas(gcf, 'report/figures/equipment_distribution.png');
    close;
    
    % 3. Failure analysis plot
    if height(processed_data.failures) > 0
        figure('Position', [100, 100, 1200, 600]);
        
        failure_types = categories(processed_data.failures.Failure_Type);
        failure_counts = countcats(processed_data.failures.Failure_Type);
        
        subplot(1, 2, 1);
        bar(failure_counts);
        set(gca, 'XTickLabel', failure_types);
        title('Failure Type Distribution');
        ylabel('Count');
        
        subplot(1, 2, 2);
        severity_counts = countcats(processed_data.failures.Severity);
        severity_names = categories(processed_data.failures.Severity);
        bar(severity_counts);
        set(gca, 'XTickLabel', severity_names);
        title('Failure Severity Distribution');
        ylabel('Count');
        
        saveas(gcf, 'report/figures/failure_analysis.png');
        close;
    end
    
    % 4. Correlation heatmap
    if isfield(analysis_results.correlations, 'correlation_matrix')
        figure('Position', [100, 100, 1000, 800]);
        
        R = analysis_results.correlations.correlation_matrix;
        var_names = analysis_results.correlations.variable_names;
        
        imagesc(R);
        colorbar;
        colormap('jet');
        caxis([-1, 1]);
        
        set(gca, 'XTick', 1:length(var_names));
        set(gca, 'YTick', 1:length(var_names));
        set(gca, 'XTickLabel', var_names);
        set(gca, 'YTickLabel', var_names);
        xtickangle(45);
        
        title('Feature Correlation Matrix');
        
        saveas(gcf, 'report/figures/correlation_heatmap.png');
        close;
    end
    
    fprintf('Analysis visualizations saved to report/figures/\n');
end