% Location: mini_projects/parallel_image_batch_processing/enhance_contrast_operation.m
% (split out of image_operations.m so it can be called from other files)

function contrast_img = enhance_contrast_operation(img)
    % Histogram-based contrast enhancement
    
    if size(img, 3) == 1
        # Grayscale
        min_val = min(img(:));
        max_val = max(img(:));
        if max_val > min_val
            contrast_img = (img - min_val) / (max_val - min_val);
        else
            contrast_img = img;
        end
    else
        # Color - process each channel
        contrast_img = zeros(size(img));
        for c = 1:size(img, 3)
            channel = img(:,:,c);
            min_val = min(channel(:));
            max_val = max(channel(:));
            if max_val > min_val
                contrast_img(:,:,c) = (channel - min_val) / (max_val - min_val);
            else
                contrast_img(:,:,c) = channel;
            end
        end
    end
end
