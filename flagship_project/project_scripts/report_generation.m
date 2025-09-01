% File location: OctaveMasterPro/flagship_project/project_scripts/report_generation.m
% Report generation module for IoT predictive maintenance project

function report_results = report_generation(processed_data, analysis_results, model_results, evaluation_results, dashboard_results)
    % Generate comprehensive project reports
    % Input: All analysis results from previous modules
    % Returns: Report generation status and file paths
    
    fprintf('Starting report generation...\n');
    
    % Initialize report results
    report_results = struct();
    
    % Create report directory structure
    create_report_directories();
    
    % Generate HTML interactive report
    fprintf('  Generating interactive HTML report...\n');
    report_results.html_report = generate_html_report(processed_data, analysis_results, model_results, evaluation_results, dashboard_results);
    
    % Generate executive summary PDF
    fprintf('  Generating executive summary...\n');
    report_results.executive_summary = generate_executive_summary(analysis_results, model_results, evaluation_results);
    
    % Generate technical appendix
    fprintf('  Generating technical appendix...\n');
    report_results.technical_appendix = generate_technical_appendix(processed_data, analysis_results, model_results, evaluation_results);
    
    % Generate model documentation
    fprintf('  Generating model documentation...\n');
    report_results.model_documentation = generate_model_documentation(model_results, evaluation_results);
    
    % Create project summary
    fprintf('  Creating project summary...\n');
    report_results.project_summary = create_project_summary(report_results);
    
    fprintf('Report generation completed\n');
    
    % Display summary
    display_report_summary(report_results);
end

function create_report_directories()
    % Create necessary report directories
    
    directories = {'report', 'report/figures', 'report/models', 'report/data'};
    
    for i = 1:length(directories)
        if ~exist(directories{i}, 'dir')
            mkdir(directories{i});
        end
    end
end

function html_report = generate_html_report(processed_data, analysis_results, model_results, evaluation_results, dashboard_results)
    % Generate comprehensive HTML report with interactive elements
    
    html_report = struct();
    html_file = 'report/analysis_report.html';
    
    % Open HTML file for writing
    fid = fopen(html_file, 'w');
    
    if fid == -1
        error('Could not create HTML report file');
    end
    
    % HTML header
    fprintf(fid, '<!DOCTYPE html>\n');
    fprintf(fid, '<html lang="en">\n<head>\n');
    fprintf(fid, '<meta charset="UTF-8">\n');
    fprintf(fid, '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n');
    fprintf(fid, '<title>IoT Predictive Maintenance Analysis Report</title>\n');
    
    % CSS styling
    write_html_styles(fid);
    
    fprintf(fid, '</head>\n<body>\n');
    
    % Report header
    write_report_header(fid, analysis_results);
    
    % Executive summary section
    write_executive_summary_html(fid, analysis_results, model_results, evaluation_results);
    
    % Data overview section
    write_data_overview_html(fid, processed_data, analysis_results);
    
    % Sensor analysis section
    write_sensor_analysis_html(fid, analysis_results);
    
    % Equipment analysis section
    write_equipment_analysis_html(fid, analysis_results);
    
    % Failure analysis section
    write_failure_analysis_html(fid, analysis_results);
    
    % Model performance section
    write_model_performance_html(fid, model_results, evaluation_results);
    
    % Visualizations section
    write_visualizations_html(fid, dashboard_results);
    
    % Recommendations section
    write_recommendations_html(fid, analysis_results, evaluation_results);
    
    % Footer
    fprintf(fid, '<footer>\n');
    fprintf(fid, '<p>Report generated on %s</p>\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, '<p>IoT Predictive Maintenance Analysis System</p>\n');
    fprintf(fid, '</footer>\n');
    
    fprintf(fid, '</body>\n</html>\n');
    fclose(fid);
    
    html_report.file_path = html_file;
    html_report.generated = true;
    html_report.size_kb = round(dir(html_file).bytes / 1024);
    
    fprintf('    HTML report saved: %s (%d KB)\n', html_file, html_report.size_kb);
end

function write_html_styles(fid)
    % Write CSS styles for HTML report
    
    fprintf(fid, '<style>\n');
    fprintf(fid, 'body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }\n');
    fprintf(fid, '.container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }\n');
    fprintf(fid, 'h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }\n');
    fprintf(fid, 'h2 { color: #34495e; margin-top: 30px; }\n');
    fprintf(fid, 'h3 { color: #7f8c8d; }\n');
    fprintf(fid, '.metric-box { display: inline-block; background: #ecf0f1; padding: 15px; margin: 10px; border-radius: 5px; min-width: 150px; text-align: center; }\n');
    fprintf(fid, '.metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }\n');
    fprintf(fid, '.metric-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }\n');
    fprintf(fid, '.alert-high { background-color: #e74c3c; color: white; }\n');
    fprintf(fid, '.alert-medium { background-color: #f39c12; color: white; }\n');
    fprintf(fid, '.alert-low { background-color: #27ae60; color: white; }\n');
    fprintf(fid, 'table { width: 100%%; border-collapse: collapse; margin: 20px 0; }\n');
    fprintf(fid, 'th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }\n');
    fprintf(fid, 'th { background-color: #3498db; color: white; }\n');
    fprintf(fid, '.progress-bar { width: 100%%; background-color: #ecf0f1; border-radius: 25px; overflow: hidden; }\n');
    fprintf(fid, '.progress-fill { height: 30px; background: linear-gradient(90deg, #e74c3c 0%%, #f39c12 50%%, #27ae60 100%%); transition: width 0.3s ease; }\n');
    fprintf(fid, 'footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 12px; }\n');
    fprintf(fid, '</style>\n');
end

function write_report_header(fid, analysis_results)
    % Write report header
    
    fprintf(fid, '<div class="container">\n');
    fprintf(fid, '<h1>IoT Predictive Maintenance Analysis Report</h1>\n');
    fprintf(fid, '<p><strong>Analysis Period:</strong> %s to %s</p>\n', ...
            datestr(analysis_results.data_summary.date_range.start, 'yyyy-mm-dd'), ...
            datestr(analysis_results.data_summary.date_range.end, 'yyyy-mm-dd'));
    fprintf(fid, '<p><strong>Total Analysis Period:</strong> %d days</p>\n', analysis_results.data_summary.analysis_period_days);
    fprintf(fid, '<p><strong>Report Generated:</strong> %s</p>\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
end

function write_executive_summary_html(fid, analysis_results, model_results, evaluation_results)
    % Write executive summary section
    
    fprintf(fid, '<h2>Executive Summary</h2>\n');
    
    % Key metrics boxes
    fprintf(fid, '<div class="metric-box">\n');
    fprintf(fid, '<div class="metric-value">%d</div>\n', analysis_results.data_summary.equipment_count);
    fprintf(fid, '<div class="metric-label">Equipment Units</div>\n');
    fprintf(fid, '</div>\n');
    
    fprintf(fid, '<div class="metric-box">\n');
    fprintf(fid, '<div class="metric-value">%d</div>\n', analysis_results.data_summary.failure_events);
    fprintf(fid, '<div class="metric-label">Failure Events</div>\n');
    fprintf(fid, '</div>\n');
    
    fprintf(fid, '<div class="metric-box">\n');
    fprintf(fid, '<div class="metric-value">%.1f%%</div>\n', analysis_results.data_summary.data_quality.sensor_completeness);
    fprintf(fid, '<div class="metric-label">Data Quality</div>\n');
    fprintf(fid, '</div>\n');
    
    if isfield(evaluation_results, 'model_comparison') && isfield(evaluation_results.model_comparison, 'best_7_day')
        fprintf(fid, '<div class="metric-box">\n');
        fprintf(fid, '<div class="metric-value">%.1f%%</div>\n', evaluation_results.model_comparison.best_7_day.accuracy * 100);
        fprintf(fid, '<div class="metric-label">Model Accuracy</div>\n');
        fprintf(fid, '</div>\n');
    end
    
    % Summary text
    failure_rate = (analysis_results.data_summary.failure_events / analysis_results.data_summary.equipment_count) * 100;
    
    fprintf(fid, '<p>This analysis covers <strong>%d equipment units</strong> monitored over <strong>%d days</strong>. ', ...
            analysis_results.data_summary.equipment_count, analysis_results.data_summary.analysis_period_days);
    fprintf(fid, 'The overall equipment failure rate is <strong>%.1f%%</strong>, with data quality at <strong>%.1f%%</strong> completeness.</p>\n', ...
            failure_rate, analysis_results.data_summary.data_quality.sensor_completeness);
    
    if isfield(evaluation_results, 'model_comparison')
        if isfield(evaluation_results.model_comparison, 'summary')
            fprintf(fid, '<p><strong>Model Performance:</strong> %s</p>\n', evaluation_results.model_comparison.summary);
        end
    end
end

function write_data_overview_html(fid, processed_data, analysis_results)
    % Write data overview section
    
    fprintf(fid, '<h2>Data Overview</h2>\n');
    
    fprintf(fid, '<h3>Dataset Summary</h3>\n');
    fprintf(fid, '<table>\n');
    fprintf(fid, '<tr><th>Dataset</th><th>Records</th><th>Date Range</th></tr>\n');
    fprintf(fid, '<tr><td>Sensor Data</td><td>%s</td><td>%s - %s</td></tr>\n', ...
            format_number(analysis_results.data_summary.sensor_records), ...
            datestr(min(processed_data.sensors.Timestamp), 'yyyy-mm-dd'), ...
            datestr(max(processed_data.sensors.Timestamp), 'yyyy-mm-dd'));
    fprintf(fid, '<tr><td>Equipment</td><td>%d</td><td>Active monitoring</td></tr>\n', analysis_results.data_summary.equipment_count);
    fprintf(fid, '<tr><td>Maintenance</td><td>%d</td><td>Historical records</td></tr>\n', analysis_results.data_summary.maintenance_records);
    fprintf(fid, '<tr><td>Failures</td><td>%d</td><td>Incident reports</td></tr>\n', analysis_results.data_summary.failure_events);
    fprintf(fid, '</table>\n');
    
    % Equipment distribution
    fprintf(fid, '<h3>Equipment Distribution</h3>\n');
    if isfield(analysis_results.data_summary, 'equipment_distribution')
        equipment_dist = analysis_results.data_summary.equipment_distribution;
        dist_fields = fieldnames(equipment_dist);
        
        fprintf(fid, '<table>\n');
        fprintf(fid, '<tr><th>Equipment Type</th><th>Count</th><th>Percentage</th></tr>\n');
        
        total_equipment = analysis_results.data_summary.equipment_count;
        for i = 1:length(dist_fields)
            count = equipment_dist.(dist_fields{i});
            percentage = (count / total_equipment) * 100;
            fprintf(fid, '<tr><td>%s</td><td>%d</td><td>%.1f%%</td></tr>\n', ...
                    strrep(dist_fields{i}, '_', ' '), count, percentage);
        end
        
        fprintf(fid, '</table>\n');
    end
end

function write_sensor_analysis_html(fid, analysis_results)
    % Write sensor analysis section
    
    fprintf(fid, '<h2>Sensor Analysis</h2>\n');
    
    if isfield(analysis_results, 'sensor_analysis')
        sensor_stats = analysis_results.sensor_analysis.statistics;
        
        fprintf(fid, '<h3>Sensor Statistics</h3>\n');
        fprintf(fid, '<table>\n');
        fprintf(fid, '<tr><th>Sensor Type</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Unit</th></tr>\n');
        
        sensors = {'Temperature', 'Pressure', 'Humidity', 'Vibration'};
        units = {'°C', 'Pa', '%%', 'g'};
        
        for i = 1:length(sensors)
            if isfield(sensor_stats, sensors{i})
                stats = sensor_stats.(sensors{i});
                fprintf(fid, '<tr><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%s</td></tr>\n', ...
                        sensors{i}, stats.mean, stats.std, stats.min, stats.max, units{i});
            end
        end
        
        fprintf(fid, '</table>\n');
        
        % Data quality assessment
        fprintf(fid, '<h3>Data Quality Assessment</h3>\n');
        fprintf(fid, '<p>Overall sensor data completeness: ');
        
        completeness = analysis_results.data_summary.data_quality.sensor_completeness;
        if completeness >= 95
            fprintf(fid, '<span class="alert-low">%.1f%% (Excellent)</span></p>\n', completeness);
        elseif completeness >= 85
            fprintf(fid, '<span class="alert-medium">%.1f%% (Good)</span></p>\n', completeness);
        else
            fprintf(fid, '<span class="alert-high">%.1f%% (Needs Attention)</span></p>\n', completeness);
        end
    end
end

function write_equipment_analysis_html(fid, analysis_results)
    % Write equipment analysis section
    
    fprintf(fid, '<h2>Equipment Analysis</h2>\n');
    
    if isfield(analysis_results, 'equipment_analysis')
        eq_analysis = analysis_results.equipment_analysis;
        
        % Age statistics
        fprintf(fid, '<h3>Equipment Age Statistics</h3>\n');
        fprintf(fid, '<p>Mean Age: <strong>%.0f days</strong> (%.1f years)</p>\n', ...
                eq_analysis.age_stats.mean_age_days, eq_analysis.age_stats.mean_age_days/365);
        fprintf(fid, '<p>Age Range: <strong>%.0f - %.0f days</strong></p>\n', ...
                eq_analysis.age_stats.min_age_days, eq_analysis.age_stats.max_age_days);
        
        % Equipment type analysis
        if isfield(eq_analysis, 'by_type')
            fprintf(fid, '<h3>Performance by Equipment Type</h3>\n');
            fprintf(fid, '<table>\n');
            fprintf(fid, '<tr><th>Type</th><th>Count</th><th>Avg Age (days)</th><th>Avg Usage (hrs)</th><th>Failure Rate</th></tr>\n');
            
            type_fields = fieldnames(eq_analysis.by_type);
            for i = 1:length(type_fields)
                type_data = eq_analysis.by_type.(type_fields{i});
                fprintf(fid, '<tr><td>%s</td><td>%d</td><td>%.0f</td><td>%.0f</td><td>%.3f</td></tr>\n', ...
                        strrep(type_fields{i}, '_', ' '), type_data.count, ...
                        type_data.avg_age, type_data.avg_operational_hours, type_data.avg_failure_rate);
            end
            
            fprintf(fid, '</table>\n');
        end
    end
end

function write_failure_analysis_html(fid, analysis_results)
    % Write failure analysis section
    
    fprintf(fid, '<h2>Failure Analysis</h2>\n');
    
    if isfield(analysis_results, 'failure_analysis')
        failure_analysis = analysis_results.failure_analysis;
        
        fprintf(fid, '<p>Total failure events analyzed: <strong>%d</strong></p>\n', failure_analysis.total_failures);
        
        % Failure types
        if isfield(failure_analysis, 'by_type')
            fprintf(fid, '<h3>Failure Types</h3>\n');
            fprintf(fid, '<table>\n');
            fprintf(fid, '<tr><th>Failure Type</th><th>Count</th><th>Percentage</th><th>Avg Downtime (hrs)</th><th>Total Cost</th></tr>\n');
            
            type_fields = fieldnames(failure_analysis.by_type);
            for i = 1:length(type_fields)
                type_data = failure_analysis.by_type.(type_fields{i});
                fprintf(fid, '<tr><td>%s</td><td>%d</td><td>%.1f%%</td><td>%.1f</td><td>$%s</td></tr>\n', ...
                        strrep(type_fields{i}, '_', ' '), type_data.count, type_data.percentage, ...
                        type_data.avg_downtime, format_number(type_data.total_repair_cost));
            end
            
            fprintf(fid, '</table>\n');
        end
        
        % Overall impact
        if isfield(failure_analysis, 'cost_impact')
            cost_impact = failure_analysis.cost_impact;
            fprintf(fid, '<h3>Business Impact</h3>\n');
            fprintf(fid, '<p>Total repair costs: <strong>$%s</strong></p>\n', format_number(cost_impact.total_repair_cost));
            fprintf(fid, '<p>Total downtime: <strong>%.1f hours</strong></p>\n', cost_impact.total_downtime_hours);
            fprintf(fid, '<p>Average cost per failure: <strong>$%s</strong></p>\n', format_number(cost_impact.avg_repair_cost));
        end
    end
end

function write_model_performance_html(fid, model_results, evaluation_results)
    % Write model performance section
    
    fprintf(fid, '<h2>Predictive Model Performance</h2>\n');
    
    if isfield(model_results, 'error')
        fprintf(fid, '<p class="alert-high">Model training encountered issues: %s</p>\n', model_results.error);
        return;
    end
    
    if isfield(evaluation_results, 'model_comparison')
        comparison = evaluation_results.model_comparison;
        
        % Best model summary
        if isfield(comparison, 'best_7_day')
            best_7 = comparison.best_7_day;
            fprintf(fid, '<h3>7-Day Prediction Results</h3>\n');
            fprintf(fid, '<p>Best performing model: <strong>%s</strong></p>\n', best_7.model);
            
            fprintf(fid, '<div class="metric-box">\n');
            fprintf(fid, '<div class="metric-value">%.1f%%</div>\n', best_7.accuracy * 100);
            fprintf(fid, '<div class="metric-label">Accuracy</div>\n');
            fprintf(fid, '</div>\n');
            
            fprintf(fid, '<div class="metric-box">\n');
            fprintf(fid, '<div class="metric-value">%.3f</div>\n', best_7.f1_score);
            fprintf(fid, '<div class="metric-label">F1 Score</div>\n');
            fprintf(fid, '</div>\n');
            
            fprintf(fid, '<div class="metric-box">\n');
            fprintf(fid, '<div class="metric-value">%.3f</div>\n', best_7.roc_auc);
            fprintf(fid, '<div class="metric-label">ROC AUC</div>\n');
            fprintf(fid, '</div>\n');
        end
        
        % Model comparison table
        if isfield(evaluation_results, 'metrics_7_day')
            fprintf(fid, '<h3>Model Comparison</h3>\n');
            fprintf(fid, '<table>\n');
            fprintf(fid, '<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>ROC AUC</th></tr>\n');
            
            model_names = fieldnames(evaluation_results.metrics_7_day);
            for i = 1:length(model_names)
                model_metrics = evaluation_results.metrics_7_day.(model_names{i});
                if isfield(model_metrics, 'accuracy')
                    fprintf(fid, '<tr><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>\n', ...
                            model_names{i}, model_metrics.accuracy, model_metrics.precision, ...
                            model_metrics.recall, model_metrics.f1_score, model_metrics.roc_auc);
                end
            end
            
            fprintf(fid, '</table>\n');
        end
        
        % Business impact
        if isfield(evaluation_results, 'business_impact') && isfield(evaluation_results.business_impact, 'model_7_day')
            impact = evaluation_results.business_impact.model_7_day;
            fprintf(fid, '<h3>Business Impact</h3>\n');
            fprintf(fid, '<p>Prevented failures: <strong>%d</strong></p>\n', impact.prevented_failures);
            fprintf(fid, '<p>Estimated cost savings: <strong>$%s</strong></p>\n', format_number(impact.cost_savings));
            fprintf(fid, '<p>ROI: <strong>%.1f%%</strong></p>\n', impact.roi_percentage);
        end
    end
end

function write_visualizations_html(fid, dashboard_results)
    % Write visualizations section
    
    fprintf(fid, '<h2>Data Visualizations</h2>\n');
    
    % Include generated charts
    chart_files = {'dashboard_overview.png', 'sensor_trends.png', 'equipment_distribution.png', ...
                  'failure_analysis.png', 'performance_metrics.png', 'comprehensive_dashboard.png'};
    
    chart_titles = {'Dashboard Overview', 'Sensor Trends', 'Equipment Distribution', ...
                   'Failure Analysis', 'Performance Metrics', 'Comprehensive Dashboard'};
    
    for i = 1:length(chart_files)
        chart_path = sprintf('figures/%s', chart_files{i});
        if exist(sprintf('report/%s', chart_path), 'file')
            fprintf(fid, '<h3>%s</h3>\n', chart_titles{i});
            fprintf(fid, '<img src="%s" alt="%s" style="max-width: 100%%; height: auto; border: 1px solid #ddd; margin: 10px 0;">\n', ...
                    chart_path, chart_titles{i});
        end
    end
end

function write_recommendations_html(fid, analysis_results, evaluation_results)
    % Write recommendations section
    
    fprintf(fid, '<h2>Recommendations</h2>\n');
    
    fprintf(fid, '<h3>Immediate Actions</h3>\n');
    fprintf(fid, '<ul>\n');
    
    % Data quality recommendations
    if analysis_results.data_summary.data_quality.sensor_completeness < 90
        fprintf(fid, '<li><strong>Improve sensor data quality:</strong> Current completeness is %.1f%%. Investigate and repair sensors with poor data quality.</li>\n', ...
                analysis_results.data_summary.data_quality.sensor_completeness);
    end
    
    % Equipment recommendations
    if analysis_results.data_summary.failure_events > 0
        failure_rate = (analysis_results.data_summary.failure_events / analysis_results.data_summary.equipment_count) * 100;
        if failure_rate > 10
            fprintf(fid, '<li><strong>Address high failure rate:</strong> Current failure rate is %.1f%%. Focus on preventive maintenance for high-risk equipment.</li>\n', failure_rate);
        end
    end
    
    % Model recommendations
    if isfield(evaluation_results, 'model_comparison') && isfield(evaluation_results.model_comparison, 'best_7_day')
        best_accuracy = evaluation_results.model_comparison.best_7_day.accuracy;
        if best_accuracy < 0.8
            fprintf(fid, '<li><strong>Improve model performance:</strong> Current best model accuracy is %.1f%%. Consider collecting more data or feature engineering.</li>\n', best_accuracy * 100);
        else
            fprintf(fid, '<li><strong>Deploy predictive model:</strong> Model shows good performance (%.1f%% accuracy). Consider production deployment.</li>\n', best_accuracy * 100);
        end
    end
    
    fprintf(fid, '</ul>\n');
    
    fprintf(fid, '<h3>Strategic Initiatives</h3>\n');
    fprintf(fid, '<ul>\n');
    fprintf(fid, '<li>Implement real-time monitoring dashboard for continuous equipment health assessment</li>\n');
    fprintf(fid, '<li>Develop automated alert system based on predictive model outputs</li>\n');
    fprintf(fid, '<li>Establish predictive maintenance scheduling based on model predictions</li>\n');
    fprintf(fid, '<li>Integrate IoT data with existing maintenance management systems</li>\n');
    fprintf(fid, '<li>Train maintenance staff on predictive analytics tools and interpretation</li>\n');
    fprintf(fid, '<li>Expand sensor coverage to include additional equipment parameters</li>\n');
    fprintf(fid, '</ul>\n');
end

function executive_summary = generate_executive_summary(analysis_results, model_results, evaluation_results)
    % Generate executive summary document
    
    executive_summary = struct();
    summary_file = 'report/executive_summary.md';
    
    fid = fopen(summary_file, 'w');
    
    if fid == -1
        error('Could not create executive summary file');
    end
    
    fprintf(fid, '# IoT Predictive Maintenance - Executive Summary\n\n');
    fprintf(fid, '**Generated:** %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    
    % Key findings
    fprintf(fid, '## Key Findings\n\n');
    
    failure_rate = (analysis_results.data_summary.failure_events / analysis_results.data_summary.equipment_count) * 100;
    fprintf(fid, '- **Equipment Portfolio:** %d units monitored over %d days\n', ...
            analysis_results.data_summary.equipment_count, analysis_results.data_summary.analysis_period_days);
    fprintf(fid, '- **Failure Rate:** %.1f%% of equipment experienced failures\n', failure_rate);
    fprintf(fid, '- **Data Quality:** %.1f%% sensor data completeness\n', analysis_results.data_summary.data_quality.sensor_completeness);
    
    if isfield(evaluation_results, 'model_comparison') && isfield(evaluation_results.model_comparison, 'best_7_day')
        fprintf(fid, '- **Predictive Accuracy:** %.1f%% for 7-day failure prediction\n', ...
                evaluation_results.model_comparison.best_7_day.accuracy * 100);
    end
    
    % Business impact
    fprintf(fid, '\n## Business Impact\n\n');
    
    if isfield(evaluation_results, 'business_impact') && isfield(evaluation_results.business_impact, 'model_7_day')
        impact = evaluation_results.business_impact.model_7_day;
        fprintf(fid, '- **Cost Savings:** $%s annually from predictive maintenance\n', format_number(impact.cost_savings));
        fprintf(fid, '- **ROI:** %.1f%% return on investment\n', impact.roi_percentage);
        fprintf(fid, '- **Prevented Failures:** %d failures successfully predicted\n', impact.prevented_failures);
    else
        fprintf(fid, '- **Potential Savings:** Estimated 20-30%% reduction in unplanned downtime\n');
        fprintf(fid, '- **Maintenance Efficiency:** Improved scheduling and resource allocation\n');
    end
    
    % Recommendations
    fprintf(fid, '\n## Strategic Recommendations\n\n');
    fprintf(fid, '1. **Deploy Predictive Models:** Implement real-time failure prediction system\n');
    fprintf(fid, '2. **Enhance Data Collection:** Improve sensor coverage and data quality\n');
    fprintf(fid, '3. **Integrate Systems:** Connect predictive analytics with maintenance workflows\n');
    fprintf(fid, '4. **Staff Training:** Develop competencies in predictive maintenance\n');
    fprintf(fid, '5. **Continuous Improvement:** Establish feedback loops for model refinement\n');
    
    % Next steps
    fprintf(fid, '\n## Immediate Next Steps\n\n');
    fprintf(fid, '- [ ] Review and approve predictive maintenance strategy\n');
    fprintf(fid, '- [ ] Allocate resources for system implementation\n');
    fprintf(fid, '- [ ] Define KPIs and success metrics\n');
    fprintf(fid, '- [ ] Plan pilot deployment for high-priority equipment\n');
    fprintf(fid, '- [ ] Schedule staff training and change management\n');
    
    fclose(fid);
    
    executive_summary.file_path = summary_file;
    executive_summary.generated = true;
    
    fprintf('    Executive summary saved: %s\n', summary_file);
end

function technical_appendix = generate_technical_appendix(processed_data, analysis_results, model_results, evaluation_results)
    % Generate detailed technical appendix
    
    technical_appendix = struct();
    appendix_file = 'report/technical_appendix.md';
    
    fid = fopen(appendix_file, 'w');
    
    if fid == -1
        error('Could not create technical appendix file');
    end
    
    fprintf(fid, '# Technical Appendix - IoT Predictive Maintenance Analysis\n\n');
    
    % Data preprocessing details
    fprintf(fid, '## Data Preprocessing\n\n');
    fprintf(fid, '### Data Sources\n\n');
    fprintf(fid, '| Dataset | Records | Features | Date Range |\n');
    fprintf(fid, '|---------|---------|----------|-----------|\n');
    fprintf(fid, '| Sensor Data | %s | 4 | %s to %s |\n', ...
            format_number(analysis_results.data_summary.sensor_records), ...
            datestr(analysis_results.data_summary.date_range.start, 'yyyy-mm-dd'), ...
            datestr(analysis_results.data_summary.date_range.end, 'yyyy-mm-dd'));
    fprintf(fid, '| Equipment | %d | 8 | Active monitoring |\n', analysis_results.data_summary.equipment_count);
    fprintf(fid, '| Maintenance | %d | 6 | Historical records |\n', analysis_results.data_summary.maintenance_records);
    fprintf(fid, '| Failures | %d | 7 | Incident reports |\n', analysis_results.data_summary.failure_events);
    
    fprintf(fid, '\n### Data Quality Metrics\n\n');
    fprintf(fid, '- **Sensor Completeness:** %.1f%%\n', analysis_results.data_summary.data_quality.sensor_completeness);
    fprintf(fid, '- **Missing Value Treatment:** Linear interpolation by sensor\n');
    fprintf(fid, '- **Outlier Detection:** 3-sigma rule with percentile capping\n');
    fprintf(fid, '- **Feature Engineering:** Rolling statistics, trend analysis, composite indices\n');
    
    % Model methodology
    fprintf(fid, '\n## Modeling Methodology\n\n');
    fprintf(fid, '### Feature Engineering\n\n');
    fprintf(fid, '- **Temporal Features:** Rolling means, standard deviations, trends\n');
    fprintf(fid, '- **Equipment Features:** Age, usage hours, maintenance efficiency\n');
    fprintf(fid, '- **Sensor Features:** Multi-sensor composite indices, peak detection\n');
    fprintf(fid, '- **Maintenance Features:** Days since last maintenance, maintenance type\n');
    
    fprintf(fid, '\n### Model Selection\n\n');
    if isfield(model_results, 'models_7_day')
        model_names = fieldnames(model_results.models_7_day);
        fprintf(fid, '**Algorithms Evaluated:**\n');
        for i = 1:length(model_names)
            if ~isfield(model_results.models_7_day.(model_names{i}), 'error')
                fprintf(fid, '- %s\n', strrep(model_names{i}, '_', ' '));
            end
        end
    end
    
    fprintf(fid, '\n**Cross-Validation:** 5-fold stratified sampling\n');
    fprintf(fid, '**Performance Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC\n');
    
    % Results details
    if isfield(evaluation_results, 'metrics_7_day')
        fprintf(fid, '\n### Detailed Results\n\n');
        fprintf(fid, '| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n');
        fprintf(fid, '|-------|----------|-----------|--------|----------|---------|\n');
        
        model_names = fieldnames(evaluation_results.metrics_7_day);
        for i = 1:length(model_names)
            metrics = evaluation_results.metrics_7_day.(model_names{i});
            if isfield(metrics, 'accuracy')
                fprintf(fid, '| %s | %.3f | %.3f | %.3f | %.3f | %.3f |\n', ...
                        model_names{i}, metrics.accuracy, metrics.precision, ...
                        metrics.recall, metrics.f1_score, metrics.roc_auc);
            end
        end
    end
    
    % Implementation notes
    fprintf(fid, '\n## Implementation Notes\n\n');
    fprintf(fid, '### System Requirements\n\n');
    fprintf(fid, '- **Computing Platform:** MATLAB/Octave\n');
    fprintf(fid, '- **Memory Requirements:** Minimum 4GB RAM\n');
    fprintf(fid, '- **Storage:** 1GB for data and models\n');
    fprintf(fid, '- **Processing Time:** ~30 minutes for full analysis\n');
    
    fprintf(fid, '\n### Model Deployment\n\n');
    fprintf(fid, '- **Real-time Scoring:** Models saved in .mat format\n');
    fprintf(fid, '- **Feature Pipeline:** Automated preprocessing required\n');
    fprintf(fid, '- **Update Frequency:** Recommended monthly retraining\n');
    fprintf(fid, '- **Monitoring:** Track prediction accuracy and data drift\n');
    
    fprintf(fid, '\n### Limitations\n\n');
    fprintf(fid, '- **Data Dependency:** Model performance depends on data quality\n');
    fprintf(fid, '- **Domain Specificity:** Models trained for current equipment types\n');
    fprintf(fid, '- **Temporal Validity:** Performance may degrade over time\n');
    fprintf(fid, '- **External Factors:** Weather, operational changes not captured\n');
    
    fclose(fid);
    
    technical_appendix.file_path = appendix_file;
    technical_appendix.generated = true;
    
    fprintf('    Technical appendix saved: %s\n', appendix_file);
end

function model_docs = generate_model_documentation(model_results, evaluation_results)
    % Generate model documentation
    
    model_docs = struct();
    docs_file = 'report/model_documentation.md';
    
    fid = fopen(docs_file, 'w');
    
    if fid == -1
        error('Could not create model documentation file');
    end
    
    fprintf(fid, '# Model Documentation\n\n');
    fprintf(fid, '## Model Overview\n\n');
    
    if isfield(evaluation_results, 'model_comparison') && isfield(evaluation_results.model_comparison, 'best_7_day')
        best_model = evaluation_results.model_comparison.best_7_day;
        fprintf(fid, '**Best Performing Model:** %s\n\n', best_model.model);
        fprintf(fid, '- **Accuracy:** %.3f\n', best_model.accuracy);
        fprintf(fid, '- **F1 Score:** %.3f\n', best_model.f1_score);
        fprintf(fid, '- **ROC AUC:** %.3f\n', best_model.roc_auc);
    end
    
    fprintf(fid, '\n## Feature Importance\n\n');
    if isfield(model_results, 'feature_importance_7_day') && isfield(model_results.feature_importance_7_day, 'logistic')
        importance_map = model_results.feature_importance_7_day.logistic;
        features = keys(importance_map);
        
        fprintf(fid, '| Feature | Importance Score |\n');
        fprintf(fid, '|---------|------------------|\n');
        
        % Sort features by importance
        importance_values = zeros(length(features), 1);
        for i = 1:length(features)
            importance_values(i) = importance_map(features{i});
        end
        
        [sorted_importance, sort_idx] = sort(importance_values, 'descend');
        sorted_features = features(sort_idx);
        
        for i = 1:min(10, length(sorted_features))
            fprintf(fid, '| %s | %.4f |\n', sorted_features{i}, sorted_importance(i));
        end
    end
    
    fprintf(fid, '\n## Model Usage\n\n');
    fprintf(fid, '### Loading the Model\n\n');
    fprintf(fid, '```matlab\n');
    fprintf(fid, 'load(''report/models/predictive_model.mat'');\n');
    fprintf(fid, 'best_model = model_results.models_7_day.logistic;\n');
    fprintf(fid, '```\n\n');
    
    fprintf(fid, '### Making Predictions\n\n');
    fprintf(fid, '```matlab\n');
    fprintf(fid, '%% Prepare new data (same preprocessing as training)\n');
    fprintf(fid, 'X_new = preprocess_new_data(raw_sensor_data);\n');
    fprintf(fid, '\n');
    fprintf(fid, '%% Make predictions\n');
    fprintf(fid, 'X_with_bias = [ones(size(X_new, 1), 1), X_new];\n');
    fprintf(fid, 'z = X_with_bias * best_model.coefficients;\n');
    fprintf(fid, 'probabilities = 1 ./ (1 + exp(-z));\n');
    fprintf(fid, 'predictions = double(probabilities > 0.5);\n');
    fprintf(fid, '```\n\n');
    
    fprintf(fid, '### Model Maintenance\n\n');
    fprintf(fid, '- **Retraining Frequency:** Monthly or when accuracy drops below 0.8\n');
    fprintf(fid, '- **Data Requirements:** Minimum 1000 samples with recent failure events\n');
    fprintf(fid, '- **Monitoring:** Track prediction accuracy on holdout set\n');
    fprintf(fid, '- **Version Control:** Maintain model versioning for rollback capability\n');
    
    fclose(fid);
    
    model_docs.file_path = docs_file;
    model_docs.generated = true;
    
    fprintf('    Model documentation saved: %s\n', docs_file);
end

function project_summary = create_project_summary(report_results)
    % Create overall project summary
    
    project_summary = struct();
    
    % Count generated files
    generated_files = 0;
    total_files = 0;
    
    fields = fieldnames(report_results);
    for i = 1:length(fields)
        if isstruct(report_results.(fields{i})) && isfield(report_results.(fields{i}), 'generated')
            total_files = total_files + 1;
            if report_results.(fields{i}).generated
                generated_files = generated_files + 1;
            end
        end
    end
    
    project_summary.files_generated = generated_files;
    project_summary.total_files = total_files;
    project_summary.success_rate = (generated_files / total_files) * 100;
    project_summary.completion_time = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Calculate total report size
    total_size = 0;
    if exist('report/analysis_report.html', 'file')
        total_size = total_size + dir('report/analysis_report.html').bytes;
    end
    if exist('report/executive_summary.md', 'file')
        total_size = total_size + dir('report/executive_summary.md').bytes;
    end
    if exist('report/technical_appendix.md', 'file')
        total_size = total_size + dir('report/technical_appendix.md').bytes;
    end
    
    project_summary.total_size_kb = round(total_size / 1024);
    
    fprintf('    Project summary created\n');
end

function display_report_summary(report_results)
    % Display comprehensive report summary
    
    fprintf('\n=== REPORT GENERATION SUMMARY ===\n');
    fprintf('Generated on: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf('Total files generated: %d/%d\n', report_results.project_summary.files_generated, report_results.project_summary.total_files);
    fprintf('Success rate: %.1f%%\n', report_results.project_summary.success_rate);
    fprintf('Total report size: %d KB\n', report_results.project_summary.total_size_kb);
    
    fprintf('\n--- Generated Reports ---\n');
    
    if isfield(report_results, 'html_report') && report_results.html_report.generated
        fprintf('✓ Interactive HTML Report: %s (%d KB)\n', report_results.html_report.file_path, report_results.html_report.size_kb);
    end
    
    if isfield(report_results, 'executive_summary') && report_results.executive_summary.generated
        fprintf('✓ Executive Summary: %s\n', report_results.executive_summary.file_path);
    end
    
    if isfield(report_results, 'technical_appendix') && report_results.technical_appendix.generated
        fprintf('✓ Technical Appendix: %s\n', report_results.technical_appendix.file_path);
    end
    
    if isfield(report_results, 'model_documentation') && report_results.model_documentation.generated
        fprintf('✓ Model Documentation: %s\n', report_results.model_documentation.file_path);
    end
    
    fprintf('\n--- Available Visualizations ---\n');
    
    viz_files = {'dashboard_overview.png', 'sensor_trends.png', 'equipment_distribution.png', ...
                'failure_analysis.png', 'performance_metrics.png', 'comprehensive_dashboard.png'};
    
    for i = 1:length(viz_files)
        if exist(sprintf('report/figures/%s', viz_files{i}), 'file')
            fprintf('✓ %s\n', viz_files{i});
        end
    end
    
    fprintf('\n--- Next Steps ---\n');
    fprintf('1. Review HTML report: report/analysis_report.html\n');
    fprintf('2. Share executive summary with stakeholders\n');
    fprintf('3. Deploy best performing predictive model\n');
    fprintf('4. Set up monitoring dashboard\n');
    fprintf('5. Schedule model retraining\n');
    
    fprintf('\n=== REPORT GENERATION COMPLETE ===\n');
end

function formatted_number = format_number(number)
    % Format numbers with thousands separators
    
    if number >= 1e6
        formatted_number = sprintf('%.1fM', number / 1e6);
    elseif number >= 1e3
        formatted_number = sprintf('%.1fK', number / 1e3);
    else
        formatted_number = sprintf('%.0f', number);
    end
end