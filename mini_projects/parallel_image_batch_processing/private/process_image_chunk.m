% Location: mini_projects/parallel_image_batch_processing/private/process_image_chunk.m
% (split out of batch_processor.m so it can be called from other files)

function chunk_result = process_image_chunk(chunk_files, output_dir, operation_func, save_results)
    % Process a chunk of images
    
    num_files = length(chunk_files);
    num_processed = 0;
    error_count = 0;
    
    for i = 1:num_files
        try
            img = load_image_safe(chunk_files{i});
            
            if ~isempty(img)
                processed_img = operation_func(img);
                
                if save_results
                    [~, name, ~] = fileparts(chunk_files{i});
                    output_file = fullfile(output_dir, [name, '_processed.png']);
                    save_image_safe(processed_img, output_file);
                end
                
                num_processed = num_processed + 1;
            end
            
        catch err
            error_count = error_count + 1;
        end
    end
    
    chunk_result.num_processed = num_processed;
    chunk_result.error_count = error_count;
end
