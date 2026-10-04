% Location: mini_projects/parallel_image_batch_processing/private/cleanup_temp_dirs.m
% (split out of parallel_demo.m so performance_benchmark.m can call it too)

function cleanup_temp_dirs(dir_list)
    # Clean up temporary directories
    
    for i = 1:length(dir_list)
        if exist(dir_list{i}, 'dir')
            try
                rmdir(dir_list{i}, 's');
                fprintf('Cleaned up directory: %s\n', dir_list{i});
            catch
                fprintf('Could not clean up directory: %s\n', dir_list{i});
            end
        end
    end
end
