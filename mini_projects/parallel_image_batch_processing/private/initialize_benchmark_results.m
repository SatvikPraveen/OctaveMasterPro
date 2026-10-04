% Location: mini_projects/parallel_image_batch_processing/private/initialize_benchmark_results.m
% (split out of performance_benchmark.m so it can be called from other files)

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
