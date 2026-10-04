% Location: mini_projects/parallel_image_batch_processing/private/apply_simple_bilateral.m
% (split out of image_operations.m so it can be called from other files)

function filtered_img = apply_simple_bilateral(img, strength)
    % Simplified bilateral filter for batch processing
    
    if size(img, 3) > 1
        # Process each channel separately
        filtered_img = zeros(size(img));
        for c = 1:size(img, 3)
            filtered_img(:,:,c) = apply_simple_bilateral(img(:,:,c), strength);
        end
        return;
    end
    
    [h, w] = size(img);
    filtered_img = zeros(h, w);
    
    sigma_spatial = 2 * strength;
    sigma_intensity = 0.1 * strength;
    kernel_size = 2 * ceil(3 * sigma_spatial) + 1;
    pad_size = floor(kernel_size / 2);
    
    # Create spatial weights
    [x, y] = meshgrid(-pad_size:pad_size);
    spatial_weights = exp(-(x.^2 + y.^2) / (2 * sigma_spatial^2));
    
    padded_img = padarray(img, [pad_size, pad_size], 'replicate');
    
    # Simplified bilateral filtering (faster approximation)
    for i = 1:h
        for j = 1:w
            neighborhood = padded_img(i:i+kernel_size-1, j:j+kernel_size-1);
            center_value = img(i, j);
            
            intensity_diff = abs(neighborhood - center_value);
            intensity_weights = exp(-intensity_diff.^2 / (2 * sigma_intensity^2));
            
            combined_weights = spatial_weights .* intensity_weights;
            combined_weights = combined_weights / sum(combined_weights(:));
            
            filtered_img(i, j) = sum(sum(neighborhood .* combined_weights));
        end
    end
end
