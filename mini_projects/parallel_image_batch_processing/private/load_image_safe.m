% Location: mini_projects/parallel_image_batch_processing/private/load_image_safe.m
% (split out of batch_processor.m so it can be called from other files)

function img = load_image_safe(filename)
    % Safely load image with error handling
    
    try
        if exist(filename, 'file')
            img = imread(filename);
            img = double(img);
            
            # Normalize if needed
            if max(img(:)) > 1
                img = img / 255;
            end
        else
            img = [];
        end
    catch
        img = [];
    end
end
