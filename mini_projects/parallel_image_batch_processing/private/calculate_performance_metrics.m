% Location: mini_projects/parallel_image_batch_processing/private/calculate_performance_metrics.m
% (split out of performance_benchmark.m so it can be called from other files)

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
