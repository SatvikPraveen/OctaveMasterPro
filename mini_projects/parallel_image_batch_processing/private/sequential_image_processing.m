% Location: mini_projects/parallel_image_batch_processing/private/sequential_image_processing.m
% (split out of batch_processor.m so it can be called from other files)

function results = sequential_image_processing(image_files, output_dir, operation_func, show_progress, save_results)
    % Execute sequential image processing
    
    num_images = length(image_files);
    num_processed = 0;
    error_count = 0;
    memory_usage = [];
    
    fprintf('Starting sequential processing...\n');
    
    for i = 1:num_images
        if show_progress && mod(i, 10) == 0
            fprintf('Processing image %d/%d (%.1f%%)...\n', i, num_images, i/num_images*100);
        end
        
        try
            # Load image
            img = load_image_safe(image_files{i});
            
            if ~isempty(img)
                # Apply operation
                processed_img = operation_func(img);
                
                # Save result
                if save_results
                    [~, name, ~] = fileparts(image_files{i});
                    output_file = fullfile(output_dir, [name, '_processed.png']);
                    save_image_safe(processed_img, output_file);
                end
                
                num_processed = num_processed + 1;
            end
            
        catch err
            error_count = error_count + 1;
            fprintf('Error processing %s: %s\n', image_files{i}, err.message);
        end
        
        # Monitor memory usage
        if mod(i, 5) == 0
            memory_usage = [memory_usage, get_memory_usage()];
        end
    end
    
    results.num_processed = num_processed;
    results.error_count = error_count;
    results.memory_usage = memory_usage;
    
    fprintf('Sequential processing completed: %d images processed.\n', num_processed);
end
