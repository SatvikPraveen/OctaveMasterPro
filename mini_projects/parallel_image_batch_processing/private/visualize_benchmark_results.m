% Location: mini_projects/parallel_image_batch_processing/private/visualize_benchmark_results.m
% (split out of performance_benchmark.m so it can be called from other files)

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
