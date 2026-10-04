% Location: mini_projects/parallel_image_batch_processing/private/save_benchmark_results.m
% (split out of performance_benchmark.m so it can be called from other files)

function save_benchmark_results(benchmark_results, filename)
    % Save benchmark results to file
    
    try
        save(filename, 'benchmark_results');
        fprintf('Benchmark results saved to: %s\n', filename);
    catch err
        fprintf('Error saving benchmark results: %s\n', err.message);
    end
end
