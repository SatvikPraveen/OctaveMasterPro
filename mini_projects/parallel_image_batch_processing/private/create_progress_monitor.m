% Location: mini_projects/parallel_image_batch_processing/private/create_progress_monitor.m
% (split out of batch_processor.m so it can be called from other files)

function progress_monitor = create_progress_monitor(total_items)
    % Create progress monitoring structure
    
    progress_monitor.total_items = total_items;
    progress_monitor.processed_items = 0;
    progress_monitor.start_time = tic;
    progress_monitor.last_update = 0;
end
