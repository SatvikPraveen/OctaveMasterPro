% Location: mini_projects/parallel_image_batch_processing/private/save_image_safe.m
% (split out of batch_processor.m so it can be called from other files)

function success = save_image_safe(img, filename)
    % Safely save image with error handling
    
    try
        if max(img(:)) <= 1
            img = img * 255;
        end
        
        img = uint8(max(0, min(255, img)));
        imwrite(img, filename);
        success = true;
    catch
        success = false;
    end
end
