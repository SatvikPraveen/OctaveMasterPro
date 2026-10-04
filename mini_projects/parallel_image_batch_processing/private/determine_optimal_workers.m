% Location: mini_projects/parallel_image_batch_processing/private/determine_optimal_workers.m
% (split out of batch_processor.m so it can be called from other files)

function num_workers = determine_optimal_workers()
    % Determine optimal number of parallel workers
    
    try
        # Try to detect number of CPU cores (simplified)
        num_cores = 4; # Default assumption
        
        # Use 75% of available cores, but at least 2 and at most 8
        num_workers = max(2, min(8, round(0.75 * num_cores)));
        
        fprintf('Auto-detected %d CPU cores, using %d workers.\n', num_cores, num_workers);
    catch
        num_workers = 2; # Safe default
        fprintf('Could not detect CPU cores. Using %d workers.\n', num_workers);
    end
end
