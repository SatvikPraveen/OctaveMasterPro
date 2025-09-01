% Location: mini_projects/parallel_image_batch_processing/image_operations.m
% Core Image Processing Functions for Batch Processing

function processed_img = resize_operation(img, varargin)
    % Resize image operation for batch processing
    
    target_size = [256, 256];
    method = 'bilinear';
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'size', target_size = varargin{i+1};
            case 'method', method = varargin{i+1};
        end
    end
    
    [h, w, c] = size(img);
    
    if h == target_size(1) && w == target_size(2)
        processed_img = img;
        return;
    end
    
    # Simple bilinear interpolation resize
    new_h = target_size(1);
    new_w = target_size(2);
    
    [X_old, Y_old] = meshgrid(1:w, 1:h);
    [X_new, Y_new] = meshgrid(linspace(1, w, new_w), linspace(1, h, new_h));
    
    if c == 1
        processed_img = interp2(X_old, Y_old, img, X_new, Y_new, 'linear', 0);
    else
        processed_img = zeros(new_h, new_w, c);
        for ch = 1:c
            processed_img(:,:,ch) = interp2(X_old, Y_old, img(:,:,ch), X_new, Y_new, 'linear', 0);
        end
    end
end

function processed_img = filter_operation(img, varargin)
    % Apply filtering operations
    
    filter_type = 'gaussian';
    sigma = 1.5;
    kernel_size = 5;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'type', filter_type = varargin{i+1};
            case 'sigma', sigma = varargin{i+1};
            case 'kernel_size', kernel_size = varargin{i+1};
        end
    end
    
    switch lower(filter_type)
        case 'gaussian'
            processed_img = apply_gaussian_blur(img, sigma);
        case 'median'
            processed_img = apply_median_filter_fast(img, kernel_size);
        case 'sharpen'
            processed_img = apply_sharpening_filter(img);
        case 'edge'
            processed_img = apply_edge_detection(img);
        otherwise
            processed_img = img;
    end
end

function blurred_img = apply_gaussian_blur(img, sigma)
    % Fast Gaussian blur implementation
    
    kernel_size = 2 * ceil(3 * sigma) + 1;
    
    # Create 1D Gaussian kernel
    x = -floor(kernel_size/2):floor(kernel_size/2);
    gaussian_1d = exp(-x.^2 / (2*sigma^2));
    gaussian_1d = gaussian_1d / sum(gaussian_1d);
    
    # Apply separable filtering
    if size(img, 3) == 1
        blurred_img = conv2(gaussian_1d, gaussian_1d', img, 'same');
    else
        blurred_img = zeros(size(img));
        for c = 1:size(img, 3)
            blurred_img(:,:,c) = conv2(gaussian_1d, gaussian_1d', img(:,:,c), 'same');
        end
    end
end

function filtered_img = apply_median_filter_fast(img, kernel_size)
    % Fast median filter implementation
    
    [h, w, c] = size(img);
    filtered_img = zeros(size(img));
    pad_size = floor(kernel_size / 2);
    
    for ch = 1:c
        channel = img(:,:,ch);
        padded = padarray(channel, [pad_size, pad_size], 'replicate');
        
        for i = 1:h
            for j = 1:w
                window = padded(i:i+kernel_size-1, j:j+kernel_size-1);
                filtered_img(i, j, ch) = median(window(:));
            end
        end
    end
end

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

function edges = apply_edge_detection(img)
    % Simple edge detection for batch processing
    
    if size(img, 3) > 1
        gray_img = 0.299*img(:,:,1) + 0.587*img(:,:,2) + 0.114*img(:,:,3);
    else
        gray_img = img;
    end
    
    # Sobel operators
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    
    grad_x = conv2(gray_img, sobel_x, 'same');
    grad_y = conv2(gray_img, sobel_y, 'same');
    
    edges = sqrt(grad_x.^2 + grad_y.^2);
    
    # Normalize
    edges = edges / max(edges(:));
end

function enhanced_img = enhance_operation(img, varargin)
    % Comprehensive image enhancement
    
    enhance_contrast = true;
    enhance_sharpness = true;
    denoise = true;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'contrast', enhance_contrast = varargin{i+1};
            case 'sharpness', enhance_sharpness = varargin{i+1};
            case 'denoise', denoise = varargin{i+1};
        end
    end
    
    enhanced_img = img;
    
    # Denoising
    if denoise
        enhanced_img = apply_gaussian_blur(enhanced_img, 0.8);
    end
    
    # Contrast enhancement
    if enhance_contrast
        enhanced_img = enhance_contrast_operation(enhanced_img);
    end
    
    # Sharpening
    if enhance_sharpness
        enhanced_img = apply_sharpening_filter(enhanced_img);
    end
end

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

function thumbnail_img = create_thumbnail(img, varargin)
    % Create thumbnail with optional border and metadata
    
    thumb_size = [128, 128];
    add_border = true;
    border_color = [0.5, 0.5, 0.5];
    border_width = 2;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'size', thumb_size = varargin{i+1};
            case 'border', add_border = varargin{i+1};
            case 'border_color', border_color = varargin{i+1};
            case 'border_width', border_width = varargin{i+1};
        end
    end
    
    # Resize to thumbnail size
    thumbnail_img = resize_operation(img, 'size', thumb_size);
    
    # Add border if requested
    if add_border
        [h, w, c] = size(thumbnail_img);
        
        if c == 1
            # Grayscale border
            thumbnail_img(1:border_width, :) = border_color(1);
            thumbnail_img(end-border_width+1:end, :) = border_color(1);
            thumbnail_img(:, 1:border_width) = border_color(1);
            thumbnail_img(:, end-border_width+1:end) = border_color(1);
        else
            # Color border
            for ch = 1:c
                thumbnail_img(1:border_width, :, ch) = border_color(min(ch, length(border_color)));
                thumbnail_img(end-border_width+1:end, :, ch) = border_color(min(ch, length(border_color)));
                thumbnail_img(:, 1:border_width, ch) = border_color(min(ch, length(border_color)));
                thumbnail_img(:, end-border_width+1:end, ch) = border_color(min(ch, length(border_color)));
            end
        end
    end
end

function watermarked_img = add_watermark(img, varargin)
    % Add watermark to image
    
    watermark_text = 'PROCESSED';
    position = 'bottom-right';
    opacity = 0.3;
    font_size = 0.05; # Relative to image height
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'text', watermark_text = varargin{i+1};
            case 'position', position = varargin{i+1};
            case 'opacity', opacity = varargin{i+1};
            case 'font_size', font_size = varargin{i+1};
        end
    end
    
    [h, w, c] = size(img);
    watermarked_img = img;
    
    # Simple text watermark (basic implementation)
    text_height = round(h * font_size);
    text_width = length(watermark_text) * round(text_height * 0.6);
    
    # Determine position
    switch lower(position)
        case 'bottom-right'
            start_row = h - text_height - 10;
            start_col = w - text_width - 10;
        case 'bottom-left'
            start_row = h - text_height - 10;
            start_col = 10;
        case 'top-right'
            start_row = 10;
            start_col = w - text_width - 10;
        case 'top-left'
            start_row = 10;
            start_col = 10;
        case 'center'
            start_row = round(h/2 - text_height/2);
            start_col = round(w/2 - text_width/2);
    end
    
    # Ensure bounds
    start_row = max(1, min(h - text_height, start_row));
    start_col = max(1, min(w - text_width, start_col));
    
    # Create simple rectangular watermark
    end_row = min(h, start_row + text_height);
    end_col = min(w, start_col + text_width);
    
    if c == 1
        # Grayscale
        watermarked_img(start_row:end_row, start_col:end_col) = ...
            (1 - opacity) * watermarked_img(start_row:end_row, start_col:end_col) + opacity * 0.8;
    else
        # Color - add semi-transparent overlay
        for ch = 1:c
            watermarked_img(start_row:end_row, start_col:end_col, ch) = ...
                (1 - opacity) * watermarked_img(start_row:end_row, start_col:end_col, ch) + opacity * 0.8;
        end
    end
end

function normalized_img = normalize_operation(img, varargin)
    % Normalize image values
    
    target_range = [0, 1];
    method = 'minmax';
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'range', target_range = varargin{i+1};
            case 'method', method = varargin{i+1};
        end
    end
    
    switch lower(method)
        case 'minmax'
            min_val = min(img(:));
            max_val = max(img(:));
            if max_val > min_val
                normalized_img = (img - min_val) / (max_val - min_val);
                normalized_img = normalized_img * (target_range(2) - target_range(1)) + target_range(1);
            else
                normalized_img = img;
            end
            
        case 'zscore'
            mean_val = mean(img(:));
            std_val = std(img(:));
            if std_val > 0
                normalized_img = (img - mean_val) / std_val;
            else
                normalized_img = img;
            end
            
        case 'robust'
            # Use percentiles for robust normalization
            p5 = prctile(img(:), 5);
            p95 = prctile(img(:), 95);
            if p95 > p5
                normalized_img = (img - p5) / (p95 - p5);
                normalized_img = max(0, min(1, normalized_img));
            else
                normalized_img = img;
            end
    end
end

function processed_img = color_correction_operation(img, varargin)
    % Color correction and enhancement
    
    gamma = 1.0;
    brightness = 0.0;
    contrast = 1.0;
    saturation = 1.0;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'gamma', gamma = varargin{i+1};
            case 'brightness', brightness = varargin{i+1};
            case 'contrast', contrast = varargin{i+1};
            case 'saturation', saturation = varargin{i+1};
        end
    end
    
    processed_img = img;
    
    # Apply gamma correction
    if gamma ~= 1.0
        processed_img = processed_img .^ gamma;
    end
    
    # Apply brightness adjustment
    processed_img = processed_img + brightness;
    
    # Apply contrast adjustment
    if contrast ~= 1.0
        mean_val = mean(processed_img(:));
        processed_img = (processed_img - mean_val) * contrast + mean_val;
    end
    
    # Apply saturation adjustment (for color images)
    if size(processed_img, 3) == 3 && saturation ~= 1.0
        # Convert to HSV-like adjustment
        gray = 0.299*processed_img(:,:,1) + 0.587*processed_img(:,:,2) + 0.114*processed_img(:,:,3);
        
        for c = 1:3
            processed_img(:,:,c) = gray + saturation * (processed_img(:,:,c) - gray);
        end
    end
    
    # Clamp values
    if max(img(:)) <= 1
        processed_img = max(0, min(1, processed_img));
    else
        processed_img = max(0, min(255, processed_img));
    end
end

function processed_img = noise_reduction_operation(img, varargin)
    % Noise reduction for batch processing
    
    method = 'bilateral';
    strength = 1.0;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'method', method = varargin{i+1};
            case 'strength', strength = varargin{i+1};
        end
    end
    
    switch lower(method)
        case 'gaussian'
            processed_img = apply_gaussian_blur(img, 1.0 * strength);
            
        case 'median'
            kernel_size = round(3 * strength);
            if mod(kernel_size, 2) == 0, kernel_size = kernel_size + 1; end
            processed_img = apply_median_filter_fast(img, kernel_size);
            
        case 'bilateral'
            processed_img = apply_simple_bilateral(img, strength);
            
        otherwise
            processed_img = img;
    end
end

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

function composite_img = create_composite_operation(img_list, varargin)
    % Create composite image from multiple images
    
    composition_type = 'average';
    weights = [];
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'type', composition_type = varargin{i+1};
            case 'weights', weights = varargin{i+1};
        end
    end
    
    if isempty(weights)
        weights = ones(length(img_list), 1) / length(img_list);
    end
    
    # Ensure all images are same size
    reference_size = size(img_list{1});
    for i = 2:length(img_list)
        if any(size(img_list{i}) ~= reference_size)
            img_list{i} = resize_operation(img_list{i}, 'size', reference_size(1:2));
        end
    end
    
    switch lower(composition_type)
        case 'average'
            composite_img = zeros(size(img_list{1}));
            for i = 1:length(img_list)
                composite_img = composite_img + weights(i) * img_list{i};
            end
            
        case 'maximum'
            composite_img = img_list{1};
            for i = 2:length(img_list)
                composite_img = max(composite_img, img_list{i});
            end
            
        case 'minimum'
            composite_img = img_list{1};
            for i = 2:length(img_list)
                composite_img = min(composite_img, img_list{i});
            end
    end
end

function demo_image_operations()
    % Demonstrate all image operations
    
    fprintf('\n--- Image Operations Demonstration ---\n');
    
    # Create test image
    test_img = create_test_image_for_operations();
    
    # Apply different operations
    fprintf('Applying various image operations...\n');
    
    resized = resize_operation(test_img, 'size', [200, 200]);
    filtered = filter_operation(test_img, 'type', 'gaussian', 'sigma', 2);
    sharpened = filter_operation(test_img, 'type', 'sharpen');
    edges = filter_operation(test_img, 'type', 'edge');
    enhanced = enhance_operation(test_img);
    normalized = normalize_operation(test_img, 'method', 'minmax');
    color_corrected = color_correction_operation(test_img, 'gamma', 1.2, 'contrast', 1.1);
    denoised = noise_reduction_operation(test_img, 'method', 'bilateral');
    thumbnail = create_thumbnail(test_img, 'size', [100, 100]);
    watermarked = add_watermark(test_img, 'text', 'DEMO', 'opacity', 0.3);
    
    # Display results
    figure('Position', [50, 50, 1400, 1000]);
    
    operations = {test_img, resized, filtered, sharpened, edges, enhanced, ...
                 normalized, color_corrected, denoised, thumbnail, watermarked};
    titles = {'Original', 'Resized', 'Gaussian Filter', 'Sharpened', 'Edge Detection', ...
             'Enhanced', 'Normalized', 'Color Corrected', 'Denoised', 'Thumbnail', 'Watermarked'};
    
    for i = 1:length(operations)
        subplot(3, 4, i);
        imshow(operations{i}, []);
        title(titles{i});
    end
    
    sgtitle('Image Operations Demonstration');
    
    fprintf('Image operations demonstration complete.\n');
end

function test_img = create_test_image_for_operations()
    # Create comprehensive test image
    
    img_size = 300;
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # Create RGB test image
    test_img = zeros(img_size, img_size, 3);
    
    # Red channel - circles
    center1 = [100, 100];
    center2 = [200, 200];
    radius1 = 40;
    radius2 = 30;
    
    circle1 = (x - center1(1)).^2 + (y - center1(2)).^2 <= radius1^2;
    circle2 = (x - center2(1)).^2 + (y - center2(2)).^2 <= radius2^2;
    
    test_img(:,:,1) = double(circle1) * 0.8 + double(circle2) * 0.6;
    
    # Green channel - stripes
    test_img(:,:,2) = 0.5 + 0.3 * sin(2*pi*x/50);
    
    # Blue channel - gradient
    test_img(:,:,3) = (x + y) / (2 * img_size);
    
    # Add some noise
    test_img = test_img + 0.05 * randn(size(test_img));
    
    # Clamp to valid range
    test_img = max(0, min(1, test_img));
end

function p = prctile(data, percentile)
    % Calculate percentile
    sorted_data = sort(data(:));
    n = length(sorted_data);
    index = percentile/100 * (n - 1) + 1;
    
    if index == round(index)
        p = sorted_data(round(index));
    else
        lower = floor(index);
        upper = ceil(index);
        weight = index - lower;
        p = sorted_data(lower) * (1 - weight) + sorted_data(upper) * weight;
    end
end