% Location: mini_projects/parallel_image_batch_processing/private/get_image_file_list.m
% (split out of batch_processor.m so it can be called from other files)

function image_files = get_image_file_list(input_dir)
    % Get list of image files from directory
    
    image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'};
    image_files = {};
    
    if exist(input_dir, 'dir')
        for ext_idx = 1:length(image_extensions)
            pattern = fullfile(input_dir, ['*.', image_extensions{ext_idx}]);
            files = dir(pattern);
            
            for i = 1:length(files)
                image_files{end+1} = fullfile(input_dir, files(i).name);
            end
        end
    end
end
