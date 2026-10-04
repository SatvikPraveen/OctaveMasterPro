% Location: mini_projects/parallel_image_batch_processing/private/update_progress_monitor.m
% (split out of batch_processor.m so it can be called from other files)

function update_progress_monitor(progress_monitor, items_completed)
    % Update progress monitor
    
    progress_monitor.processed_items = items_completed;
    current_time = toc(progress_monitor.start_time);
    
    if current_time - progress_monitor.last_update > 1 # Update every second
        progress_percent = items_completed / progress_monitor.total_items * 100;
        
        if items_completed > 0
            estimated_total_time = current_time * progress_monitor.total_items / items_completed;
            eta = estimated_total_time - current_time;
            
            fprintf('Progress: %.1f%% (%d/%d) - ETA: %.1f seconds\n', ...
                   progress_percent, items_completed, progress_monitor.total_items, eta);
        end
        
        progress_monitor.last_update = current_time;
    end
end
