% Location: mini_projects/parallel_image_batch_processing/private/benchmark_single_operation.m
% (split out of performance_benchmark.m so it can be called from other files)

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
