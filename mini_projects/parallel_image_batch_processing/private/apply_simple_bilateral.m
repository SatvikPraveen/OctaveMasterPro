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
    
    img = double(img);
    padded_img = padarray(img, [pad_size, pad_size], 'replicate');
    
    # Simplified bilateral filtering, accumulated per kernel offset instead
    # of per pixel (same weights and result, but vectorised over the image
    # so a 512x512 RGB image takes well under a second instead of minutes)
    weighted_sum = zeros(h, w);
    weight_total = zeros(h, w);
    for dy = 0:kernel_size-1
        for dx = 0:kernel_size-1
            shifted = padded_img(1+dy:h+dy, 1+dx:w+dx);
            intensity_diff = abs(shifted - img);
            weights = spatial_weights(dy+1, dx+1) * ...
                      exp(-intensity_diff.^2 / (2 * sigma_intensity^2));
            weighted_sum = weighted_sum + weights .* shifted;
            weight_total = weight_total + weights;
        end
    end
    filtered_img = weighted_sum ./ weight_total;
end
