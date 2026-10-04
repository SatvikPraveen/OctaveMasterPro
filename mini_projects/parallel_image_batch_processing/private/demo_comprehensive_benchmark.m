% Location: mini_projects/parallel_image_batch_processing/private/demo_comprehensive_benchmark.m
% (split out of performance_benchmark.m so it can be called from other files)

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
