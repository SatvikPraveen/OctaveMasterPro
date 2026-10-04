% Location: mini_projects/parallel_image_batch_processing/private/generate_performance_report.m
% (split out of performance_benchmark.m so it can be called from other files)

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
