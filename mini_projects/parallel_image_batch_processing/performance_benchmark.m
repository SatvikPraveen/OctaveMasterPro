% Location: mini_projects/parallel_image_batch_processing/performance_benchmark.m
% Performance Testing and Benchmarking Suite

function benchmark_results = performance_benchmark(test_dir, operations, varargin)
    % Comprehensive performance benchmarking suite
    
    batch_sizes = [4, 8, 16, 32];
    worker_counts = [1, 2, 4, 6, 8];
    iterations = 3;
    save_results_flag = true;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'batch_sizes', batch_sizes = varargin{i+1};
            case 'worker_counts', worker_counts = varargin{i+1};
            case 'iterations', iterations = varargin{i+1};
            case 'save_results', save_results_flag = varargin{i+1};
        end
    end
    
    fprintf('====================================================\n');
    fprintf('    PERFORMANCE BENCHMARKING SUITE                 \n');
    fprintf('====================================================\n\n');
    
    % Setup test environment
    if isempty(test_dir) || ~exist(test_dir, 'dir')
        test_dir = 'benchmark_temp_images';
        fprintf('Generating test images...\n');
        max_batch_size = max(batch_sizes);
        test_images = generate_test_images(test_dir, max_batch_size * 2);
    else
        test_images = get_image_file_list(test_dir);
    end
    
    num_test_images = length(test_images);
    fprintf('Using %d test images.\n', num_test_images);
    
    operation_funcs = convert_operations_to_functions(operations);
    operation_names = get_operation_names(operations);
    
    benchmark_results = initialize_benchmark_results(batch_sizes, worker_counts, operation_names, iterations);
    
    fprintf('\nStarting benchmarking...\n');
    
    for batch_idx = 1:length(batch_sizes)
        batch_size = batch_sizes(batch_idx);
        fprintf('\n--- Batch size: %d ---\n', batch_size);
        
        if num_test_images >= batch_size
            batch_images = test_images(1:batch_size);
        else
            repeat_factor = ceil(batch_size / num_test_images);
            batch_images = repmat(test_images, repeat_factor, 1);
            batch_images = batch_images(1:batch_size);
        end
        
        for worker_idx = 1:length(worker_counts)
            num_workers = worker_counts(worker_idx);
            
            for op_idx = 1:length(operation_funcs)
                operation_func = operation_funcs{op_idx};
                operation_name = operation_names{op_idx};
                
                fprintf('  %s (%d workers): ', operation_name, num_workers);
                
                iteration_times = zeros(iterations, 1);
                iteration_memory = zeros(iterations, 1);
                
                for iter = 1:iterations
                    if num_workers == 1
                        [exec_time, memory_usage] = benchmark_single_operation(batch_images, operation_func, false);
                    else
                        [exec_time, memory_usage] = benchmark_single_operation(batch_images, operation_func, true, num_workers);
                    end
                    
                    iteration_times(iter) = exec_time;
                    iteration_memory(iter) = memory_usage;
                    fprintf('.');
                end
                
                benchmark_results.execution_times(batch_idx, worker_idx, op_idx, :) = iteration_times;
                benchmark_results.memory_usage(batch_idx, worker_idx, op_idx, :) = iteration_memory;
                
                mean_time = mean(iteration_times);
                std_time = std(iteration_times);
                mean_memory = mean(iteration_memory);
                
                benchmark_results.mean_times(batch_idx, worker_idx, op_idx) = mean_time;
                benchmark_results.std_times(batch_idx, worker_idx, op_idx) = std_time;
                benchmark_results.mean_memory(batch_idx, worker_idx, op_idx) = mean_memory;
                benchmark_results.throughput(batch_idx, worker_idx, op_idx) = batch_size / mean_time;
                
                fprintf(' %.2f±%.2fs (%.1f img/s)\n', mean_time, std_time, batch_size/mean_time);
            end
        end
    end
    
    % Calculate speedups and efficiency
    calculate_performance_metrics(benchmark_results);
    
    % Visualize results
    visualize_benchmark_results(benchmark_results);
    
    % Save results if requested
    if save_results_flag
        save_benchmark_results(benchmark_results, 'benchmark_results.mat');
    end
    
    fprintf('\nBenchmarking complete!\n');
    
    cleanup_temp_dirs({test_dir});
end

function [exec_time, memory_usage] = benchmark_single_operation(image_files, operation_func, use_parallel, num_workers)
    % Benchmark a single operation
    
    if nargin < 4, num_workers = 4; end
    
    start_memory = get_memory_usage();
    
    tic;
    
    if use_parallel
        # Simulate parallel processing
        chunk_size = max(1, floor(length(image_files) / num_workers));
        chunks = create_processing_chunks(image_files, chunk_size);
        
        for chunk_idx = 1:length(chunks)
            process_image_chunk(chunks{chunk_idx}, '', operation_func, false);
        end
    else
        # Sequential processing
        for i = 1:length(image_files)
            img = load_image_safe(image_files{i});
            if ~isempty(img)
                processed = operation_func(img);
            end
        end
    end
    
    exec_time = toc;
    end_memory = get_memory_usage();
    memory_usage = end_memory - start_memory;
end

function operation_funcs = convert_operations_to_functions(operations)
    % Convert operation names to function handles
    
    operation_funcs = cell(size(operations));
    
    for i = 1:length(operations)
        if ischar(operations{i}) || isstring(operations{i})
            switch lower(operations{i})
                case 'resize'
                    operation_funcs{i} = @(img) resize_operation(img, 'size', [256, 256]);
                case 'filter'
                    operation_funcs{i} = @(img) filter_operation(img, 'type', 'gaussian');
                case 'enhance'
                    operation_funcs{i} = @(img) enhance_operation(img);
                case 'edge'
                    operation_funcs{i} = @(img) filter_operation(img, 'type', 'edge');
                case 'denoise'
                    operation_funcs{i} = @(img) noise_reduction_operation(img);
                otherwise
                    operation_funcs{i} = @(img) img; % Identity operation
            end
        else
            operation_funcs{i} = operations{i};
        end
    end
end

function operation_names = get_operation_names(operations)
    % Get operation names for display
    
    operation_names = cell(size(operations));
    
    for i = 1:length(operations)
        if ischar(operations{i}) || isstring(operations{i})
            operation_names{i} = operations{i};
        else
            operation_names{i} = sprintf('Custom_%d', i);
        end
    end
end

function results = initialize_benchmark_results(batch_sizes, worker_counts, operation_names, iterations)
    % Initialize benchmark results structure
    
    num_batches = length(batch_sizes);
    num_workers = length(worker_counts);
    num_operations = length(operation_names);
    
    results.batch_sizes = batch_sizes;
    results.worker_counts = worker_counts;
    results.operation_names = operation_names;
    results.iterations = iterations;
    
    results.execution_times = NaN(num_batches, num_workers, num_operations, iterations);
    results.memory_usage = NaN(num_batches, num_workers, num_operations, iterations);
    results.mean_times = NaN(num_batches, num_workers, num_operations);
    results.std_times = NaN(num_batches, num_workers, num_operations);
    results.mean_memory = NaN(num_batches, num_workers, num_operations);
    results.throughput = NaN(num_batches, num_workers, num_operations);
    results.speedups = NaN(num_batches, num_workers, num_operations);
    results.efficiency = NaN(num_batches, num_workers, num_operations);
end

function calculate_performance_metrics(benchmark_results)
    % Calculate derived performance metrics
    
    [num_batches, num_workers, num_operations] = size(benchmark_results.mean_times);
    
    % Calculate speedups (relative to single worker)
    for batch_idx = 1:num_batches
        for op_idx = 1:num_operations
            baseline_time = benchmark_results.mean_times(batch_idx, 1, op_idx); % Single worker
            
            for worker_idx = 1:num_workers
                current_time = benchmark_results.mean_times(batch_idx, worker_idx, op_idx);
                
                if ~isnan(baseline_time) && ~isnan(current_time) && current_time > 0
                    speedup = baseline_time / current_time;
                    efficiency = speedup / benchmark_results.worker_counts(worker_idx) * 100;
                    
                    benchmark_results.speedups(batch_idx, worker_idx, op_idx) = speedup;
                    benchmark_results.efficiency(batch_idx, worker_idx, op_idx) = efficiency;
                end
            end
        end
    end
    
    fprintf('\nPerformance metrics calculated.\n');
end

function visualize_benchmark_results(benchmark_results)
    % Create comprehensive visualization of benchmark results
    
    fprintf('Generating benchmark visualizations...\n');
    
    % Main benchmark dashboard
    figure('Position', [50, 50, 1600, 1200]);
    
    # Execution time heatmap
    subplot(3, 4, 1);
    # Average across all operations for overview
    avg_times = squeeze(mean(benchmark_results.mean_times, 3));
    imagesc(avg_times);
    colorbar;
    
    set(gca, 'XTick', 1:length(benchmark_results.worker_counts));
    set(gca, 'XTickLabel', benchmark_results.worker_counts);
    set(gca, 'YTick', 1:length(benchmark_results.batch_sizes));
    set(gca, 'YTickLabel', benchmark_results.batch_sizes);
    
    xlabel('Number of Workers');
    ylabel('Batch Size');
    title('Average Execution Time (s)');
    
    # Speedup curves
    subplot(3, 4, 2);
    colors = {'b', 'r', 'g', 'm', 'c'};
    
    for batch_idx = 1:length(benchmark_results.batch_sizes)
        avg_speedup = squeeze(mean(benchmark_results.speedups(batch_idx, :, :), 3));
        plot(benchmark_results.worker_counts, avg_speedup, [colors{mod(batch_idx-1, 5)+1}, '-o'], ...
             'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
    end
    
    # Add ideal speedup line
    plot(benchmark_results.worker_counts, benchmark_results.worker_counts, 'k--', 'LineWidth', 1);
    
    legend([cellfun(@(x) sprintf('Batch %d', x), num2cell(benchmark_results.batch_sizes), 'UniformOutput', false), {'Ideal'}], ...
           'Location', 'best');
    title('Speedup vs Number of Workers');
    xlabel('Number of Workers'); ylabel('Speedup Factor'); grid on;
    
    # Throughput comparison
    subplot(3, 4, 3);
    max_throughput = squeeze(max(benchmark_results.throughput, [], 2)); % Best performance per batch/operation
    
    for op_idx = 1:length(benchmark_results.operation_names)
        plot(benchmark_results.batch_sizes, max_throughput(:, op_idx), [colors{mod(op_idx-1, 5)+1}, '-s'], ...
             'LineWidth', 2, 'MarkerSize', 8);
        hold on;
    end
    
    legend(benchmark_results.operation_names, 'Location', 'best');
    title('Peak Throughput by Operation');
    xlabel('Batch Size'); ylabel('Images/Second'); grid on;
    
    # Efficiency analysis
    subplot(3, 4, 4);
    avg_efficiency = squeeze(mean(benchmark_results.efficiency, 3));
    
    for worker_idx = 2:length(benchmark_results.worker_counts) % Skip single worker (100% efficient)
        plot(benchmark_results.batch_sizes, avg_efficiency(:, worker_idx), ...
             [colors{mod(worker_idx-2, 5)+1}, '-^'], 'LineWidth', 1.5, 'MarkerSize', 6);
        hold on;
    end
    
    legend(cellfun(@(x) sprintf('%d workers', x), num2cell(benchmark_results.worker_counts(2:end)), 'UniformOutput', false), ...
           'Location', 'best');
    title('Parallel Efficiency');
    xlabel('Batch Size'); ylabel('Efficiency (%)'); grid on;
    
    # Memory usage analysis
    subplot(3, 4, 5);
    avg_memory = squeeze(mean(benchmark_results.mean_memory, 3));
    
    surf(benchmark_results.worker_counts, benchmark_results.batch_sizes, avg_memory);
    xlabel('Number of Workers'); ylabel('Batch Size'); zlabel('Memory Usage (MB)');
    title('Memory Usage Surface');
    
    # Operation comparison
    subplot(3, 4, 6);
    # Use mid-range batch size and worker count for comparison
    mid_batch_idx = ceil(length(benchmark_results.batch_sizes) / 2);
    mid_worker_idx = ceil(length(benchmark_results.worker_counts) / 2);
    
    op_times = squeeze(benchmark_results.mean_times(mid_batch_idx, mid_worker_idx, :));
    op_throughput = squeeze(benchmark_results.throughput(mid_batch_idx, mid_worker_idx, :));
    
    [ax, h1, h2] = plotyy(1:length(operation_names), op_times, 1:length(operation_names), op_throughput);
    set(h1, 'LineStyle', '-o', 'LineWidth', 2, 'MarkerSize', 8);
    set(h2, 'LineStyle', '-s', 'LineWidth', 2, 'MarkerSize', 8);
    
    set(ax(1), 'XTick', 1:length(operation_names), 'XTickLabel', operation_names);
    set(ax(2), 'XTick', 1:length(operation_names), 'XTickLabel', operation_names);
    ylabel(ax(1), 'Execution Time (s)');
    ylabel(ax(2), 'Throughput (img/s)');
    title('Operation Performance Comparison');
    
    # Scalability trends
    subplot(3, 4, 7);
    for op_idx = 1:min(3, length(benchmark_results.operation_names)) % Show top 3 operations
        avg_speedup_by_workers = squeeze(mean(benchmark_results.speedups(:, :, op_idx), 1));
        plot(benchmark_results.worker_counts, avg_speedup_by_workers, ...
             [colors{op_idx}, '-o'], 'LineWidth', 2, 'MarkerSize', 6);
        hold on;
    end
    
    plot(benchmark_results.worker_counts, benchmark_results.worker_counts, 'k--', 'LineWidth', 1);
    legend([benchmark_results.operation_names(1:min(3, end)), {'Ideal'}], 'Location', 'best');
    title('Scalability by Operation');
    xlabel('Number of Workers'); ylabel('Average Speedup'); grid on;
    
    # Best configuration finder
    subplot(3, 4, 8);
    % Find best worker count for each batch size
    best_configs = NaN(length(benchmark_results.batch_sizes), 1);
    best_speedups = NaN(length(benchmark_results.batch_sizes), 1);
    
    for batch_idx = 1:length(benchmark_results.batch_sizes)
        avg_speedup_for_batch = squeeze(mean(benchmark_results.speedups(batch_idx, :, :), 3));
        [best_speedup, best_worker_idx] = max(avg_speedup_for_batch);
        
        best_configs(batch_idx) = benchmark_results.worker_counts(best_worker_idx);
        best_speedups(batch_idx) = best_speedup;
    end
    
    [ax, h1, h2] = plotyy(benchmark_results.batch_sizes, best_configs, benchmark_results.batch_sizes, best_speedups);
    set(h1, 'LineStyle', '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'blue');
    set(h2, 'LineStyle', '-s', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'red');
    ylabel(ax(1), 'Optimal Workers');
    ylabel(ax(2), 'Best Speedup');
    xlabel('Batch Size');
    title('Optimal Configuration');
    grid on;
    
    # Resource utilization heatmap
    subplot(3, 4, 9);
    # Simulate CPU utilization based on speedup
    cpu_utilization = squeeze(mean(benchmark_results.speedups, 3)) ./ repmat(benchmark_results.worker_counts, length(benchmark_results.batch_sizes), 1) * 100;
    cpu_utilization = min(cpu_utilization, 100);
    
    imagesc(cpu_utilization);
    colorbar;
    colormap('hot');
    
    set(gca, 'XTick', 1:length(benchmark_results.worker_counts));
    set(gca, 'XTickLabel', benchmark_results.worker_counts);
    set(gca, 'YTick', 1:length(benchmark_results.batch_sizes));
    set(gca, 'YTickLabel', benchmark_results.batch_sizes);
    
    xlabel('Number of Workers'); ylabel('Batch Size');
    title('CPU Utilization (%)');
    
    # Performance variability
    subplot(3, 4, 10);
    avg_cv = squeeze(mean(benchmark_results.std_times ./ benchmark_results.mean_times, 3)) * 100;
    
    plot(benchmark_results.worker_counts, avg_cv);
    title('Performance Variability');
    xlabel('Number of Workers'); ylabel('Coefficient of Variation (%)'); grid on;
    
    # Memory efficiency
    subplot(3, 4, 11);
    memory_per_image = squeeze(mean(benchmark_results.mean_memory, 3)) ./ repmat(benchmark_results.batch_sizes', 1, length(benchmark_results.worker_counts));
    
    plot(benchmark_results.worker_counts, memory_per_image);
    legend(cellfun(@(x) sprintf('Batch %d', x), num2cell(benchmark_results.batch_sizes), 'UniformOutput', false), ...
           'Location', 'best');
    title('Memory per Image');
    xlabel('Number of Workers'); ylabel('Memory/Image (MB)'); grid on;
    
    # Performance summary table
    subplot(3, 4, 12);
    % Create summary statistics
    best_overall_speedup = max(benchmark_results.speedups(:));
    best_overall_throughput = max(benchmark_results.throughput(:));
    avg_efficiency = mean(benchmark_results.efficiency(:), 'omitnan');
    
    summary_data = [best_overall_speedup, best_overall_throughput, avg_efficiency];
    summary_labels = {'Best Speedup', 'Peak Throughput', 'Avg Efficiency'};
    
    bar(summary_data);
    set(gca, 'XTickLabel', summary_labels);
    title('Performance Summary');
    ylabel('Values'); grid on;
    
    % Add text annotations
    for i = 1:length(summary_data)
        text(i, summary_data(i) + 0.1, sprintf('%.1f', summary_data(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    sgtitle('Comprehensive Performance Benchmark Results');
    
    % Print summary
    fprintf('\nBenchmark Summary:\n');
    fprintf('==================\n');
    fprintf('Best speedup achieved: %.2fx\n', best_overall_speedup);
    fprintf('Peak throughput: %.1f images/second\n', best_overall_throughput);
    fprintf('Average parallel efficiency: %.1f%%\n', avg_efficiency);
end

function save_benchmark_results(benchmark_results, filename)
    % Save benchmark results to file
    
    try
        save(filename, 'benchmark_results');
        fprintf('Benchmark results saved to: %s\n', filename);
    catch err
        fprintf('Error saving benchmark results: %s\n', err.message);
    end
end

function generate_performance_report(benchmark_results)
    % Generate detailed performance report
    
    fprintf('\n====================================================\n');
    fprintf('           DETAILED PERFORMANCE REPORT             \n');
    fprintf('====================================================\n\n');
    
    % Overall statistics
    fprintf('OVERALL STATISTICS:\n');
    fprintf('------------------\n');
    
    all_speedups = benchmark_results.speedups(:);
    all_efficiency = benchmark_results.efficiency(:);
    all_throughput = benchmark_results.throughput(:);
    
    fprintf('Speedup - Mean: %.2f, Max: %.2f, Min: %.2f\n', ...
           mean(all_speedups, 'omitnan'), max(all_speedups), min(all_speedups));
    fprintf('Efficiency - Mean: %.1f%%, Max: %.1f%%, Min: %.1f%%\n', ...
           mean(all_efficiency, 'omitnan'), max(all_efficiency), min(all_efficiency));
    fprintf('Throughput - Mean: %.1f img/s, Max: %.1f img/s\n', ...
           mean(all_throughput, 'omitnan'), max(all_throughput));
    
    % Best configurations
    fprintf('\nBEST CONFIGURATIONS:\n');
    fprintf('-------------------\n');
    
    [max_speedup, max_speedup_idx] = max(all_speedups);
    [batch_idx, worker_idx, op_idx] = ind2sub(size(benchmark_results.speedups), max_speedup_idx);
    
    fprintf('Best speedup: %.2fx\n', max_speedup);
    fprintf('  Configuration: %d images, %d workers, %s operation\n', ...
           benchmark_results.batch_sizes(batch_idx), benchmark_results.worker_counts(worker_idx), ...
           benchmark_results.operation_names{op_idx});
    
    [max_throughput, max_throughput_idx] = max(all_throughput);
    [batch_idx, worker_idx, op_idx] = ind2sub(size(benchmark_results.throughput), max_throughput_idx);
    
    fprintf('Best throughput: %.1f images/second\n', max_throughput);
    fprintf('  Configuration: %d images, %d workers, %s operation\n', ...
           benchmark_results.batch_sizes(batch_idx), benchmark_results.worker_counts(worker_idx), ...
           benchmark_results.operation_names{op_idx});
    
    # Recommendations
    fprintf('\nRECOMMENDATIONS:\n');
    fprintf('---------------\n');
    
    % Find sweet spot (good balance of speedup and efficiency)
    efficiency_threshold = 70; % 70% efficiency threshold
    good_configs = benchmark_results.efficiency > efficiency_threshold;
    
    if any(good_configs(:))
        good_speedups = benchmark_results.speedups;
        good_speedups(~good_configs) = NaN;
        
        [best_balanced_speedup, best_balanced_idx] = max(good_speedups(:));
        [batch_idx, worker_idx, op_idx] = ind2sub(size(good_speedups), best_balanced_idx);
        
        fprintf('Recommended configuration (best balance): %d images, %d workers\n', ...
               benchmark_results.batch_sizes(batch_idx), benchmark_results.worker_counts(worker_idx));
        fprintf('  Achieves %.2fx speedup with %.1f%% efficiency\n', ...
               best_balanced_speedup, benchmark_results.efficiency(batch_idx, worker_idx, op_idx));
    else
        fprintf('No configuration achieved >70%% efficiency. Consider optimizing algorithms.\n');
    end
    
    fprintf('\n');
end

function demo_comprehensive_benchmark()
    % Run comprehensive benchmark demonstration
    
    fprintf('\n--- Comprehensive Benchmark Demo ---\n');
    
    # Setup comprehensive test
    operations = {'resize', 'filter', 'enhance', 'denoise'};
    
    benchmark_results = performance_benchmark('', operations, ...
        'batch_sizes', [8, 16, 24], ...
        'worker_counts', [1, 2, 4], ...
        'iterations', 2);
    
    generate_performance_report(benchmark_results);
    
    fprintf('Comprehensive benchmark demonstration complete.\n');
end