% Location: mini_projects/parallel_image_batch_processing/private/get_memory_usage.m
% (split out of batch_processor.m so it can be called from other files)

function memory_mb = get_memory_usage()
    % Get current memory usage (simplified)
    
    try
        # In a real implementation, this would query system memory
        # For demonstration, we'll simulate memory usage
        memory_mb = 100 + 50 * rand(); # Simulated MB usage
    catch
        memory_mb = NaN;
    end
end
