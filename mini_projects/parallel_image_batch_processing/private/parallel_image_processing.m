% Location: mini_projects/parallel_image_batch_processing/private/parallel_image_processing.m
% (split out of batch_processor.m so it can be called from other files)

function results = parallel_image_processing(image_files, output_dir, operation_func, num_workers, chunk_size, show_progress, save_results)
    % Execute parallel image processing
    
    num_images = length(image_files);
    num_processed = 0;
    error_count = 0;
    memory_usage = [];
    
    fprintf('Starting parallel processing with %d workers...\n', num_workers);
    
    # Create chunks for parallel processing
    chunks = create_processing_chunks(image_files, chunk_size);
    num_chunks = length(chunks);
    
    if show_progress
        fprintf('Processing %d chunks...\n', num_chunks);
    end
    
    # Process chunks on worker processes (falls back to sequential)
    [chunk_results, mode] = run_chunks(chunks, output_dir, operation_func, save_results, num_workers);
    for chunk_idx = 1:num_chunks
        num_processed = num_processed + chunk_results{chunk_idx}.num_processed;
        error_count = error_count + chunk_results{chunk_idx}.error_count;
    end
    memory_usage = get_memory_usage();
    if show_progress
        fprintf('Execution mode: %s\n', mode);
    end
    
    results.num_processed = num_processed;
    results.error_count = error_count;
    results.memory_usage = memory_usage;
    results.chunk_results = chunk_results;
    results.execution_mode = mode;
    
    fprintf('Parallel processing completed: %d images processed.\n', num_processed);
end
