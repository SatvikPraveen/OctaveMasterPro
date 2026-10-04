% Location: mini_projects/parallel_image_batch_processing/private/apply_sharpening_filter.m
% (split out of image_operations.m so it can be called from other files)

function sharpened_img = apply_sharpening_filter(img)
    % Unsharp masking for sharpening
    
    # Apply Gaussian blur
    blurred = apply_gaussian_blur(img, 1.0);
    
    # Create unsharp mask
    mask = img - blurred;
    
    # Apply sharpening
    sharpened_img = img + 1.5 * mask;
    
    # Clamp values
    if max(img(:)) <= 1
        sharpened_img = max(0, min(1, sharpened_img));
    else
        sharpened_img = max(0, min(255, sharpened_img));
    end
end
