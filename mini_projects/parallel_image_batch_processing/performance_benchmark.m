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
    benchmark_results = calculate_performance_metrics(benchmark_results);
    
    % Visualize results
    visualize_benchmark_results(benchmark_results);
    
    % Save results if requested
    if save_results_flag
        save_benchmark_results(benchmark_results, 'benchmark_results.mat');
    end
    
    fprintf('\nBenchmarking complete!\n');
    
    cleanup_temp_dirs({test_dir});
end
