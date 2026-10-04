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
    
    # Process chunks (simulated parallel processing)
    chunk_results = cell(num_chunks, 1);
    
    for chunk_idx = 1:num_chunks
        if show_progress
            fprintf('Processing chunk %d/%d...\n', chunk_idx, num_chunks);
        end
        
        chunk_files = chunks{chunk_idx};
        chunk_result = process_image_chunk(chunk_files, output_dir, operation_func, save_results);
        
        chunk_results{chunk_idx} = chunk_result;
        num_processed = num_processed + chunk_result.num_processed;
        error_count = error_count + chunk_result.error_count;
        
        # Monitor memory usage
        memory_usage = [memory_usage, get_memory_usage()];
    end
    
    results.num_processed = num_processed;
    results.error_count = error_count;
    results.memory_usage = memory_usage;
    results.chunk_results = chunk_results;
    
    fprintf('Parallel processing completed: %d images processed.\n', num_processed);
end
