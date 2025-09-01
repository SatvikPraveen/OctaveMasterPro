% Location: mini_projects/image_processing_basics/basic_filters.m
% Basic Image Filtering Operations

function filtered_img = apply_gaussian_filter(img, sigma, varargin)
    % Apply Gaussian blur filter
    %
    % Inputs:
    %   img - input image
    %   sigma - standard deviation of Gaussian kernel
    %   varargin - optional parameters:
    %     'kernel_size' - size of Gaussian kernel (default: auto)
    %     'separable' - use separable filtering (default: true)
    %
    % Output:
    %   filtered_img - Gaussian filtered image
    
    % Default parameters
    kernel_size = [];
    separable = true;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'kernel_size'
                kernel_size = varargin{i+1};
            case 'separable'
                separable = varargin{i+1};
        end
    end
    
    % Auto-determine kernel size
    if isempty(kernel_size)
        kernel_size = 2 * ceil(3 * sigma) + 1; % 6*sigma + 1 rule
    end
    
    % Ensure odd kernel size
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
    end
    
    if separable
        % Use separable 1D Gaussian filters (more efficient)
        x = -floor(kernel_size/2):floor(kernel_size/2);
        gaussian_1d = exp(-x.^2 / (2*sigma^2));
        gaussian_1d = gaussian_1d / sum(gaussian_1d);
        
        % Apply separable filtering
        if size(img, 3) == 1
            % Grayscale
            filtered_img = conv2(gaussian_1d, gaussian_1d', img, 'same');
        else
            % Color - process each channel
            filtered_img = zeros(size(img));
            for c = 1:size(img, 3)
                filtered_img(:,:,c) = conv2(gaussian_1d, gaussian_1d', img(:,:,c), 'same');
            end
        end
    else
        % Use 2D Gaussian kernel
        [x, y] = meshgrid(-floor(kernel_size/2):floor(kernel_size/2));
        gaussian_2d = exp(-(x.^2 + y.^2) / (2*sigma^2));
        gaussian_2d = gaussian_2d / sum(gaussian_2d(:));
        
        % Apply 2D convolution
        if size(img, 3) == 1
            filtered_img = conv2(img, gaussian_2d, 'same');
        else
            filtered_img = zeros(size(img));
            for c = 1:size(img, 3)
                filtered_img(:,:,c) = conv2(img(:,:,c), gaussian_2d, 'same');
            end
        end
    end
end

function filtered_img = apply_median_filter(img, kernel_size)
    % Apply median filter for noise reduction
    %
    % Inputs:
    %   img - input image
    %   kernel_size - size of median filter window (odd number)
    %
    % Output:
    %   filtered_img - median filtered image
    
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
        fprintf('Kernel size adjusted to %d (must be odd)\n', kernel_size);
    end
    
    [h, w, c] = size(img);
    filtered_img = zeros(size(img));
    pad_size = floor(kernel_size / 2);
    
    % Pad image
    if c == 1
        padded_img = padarray(img, [pad_size, pad_size], 'replicate');
    else
        padded_img = zeros(h + 2*pad_size, w + 2*pad_size, c);
        for ch = 1:c
            padded_img(:,:,ch) = padarray(img(:,:,ch), [pad_size, pad_size], 'replicate');
        end
    end
    
    % Apply median filter
    for i = 1:h
        for j = 1:w
            for ch = 1:c
                window = padded_img(i:i+kernel_size-1, j:j+kernel_size-1, ch);
                filtered_img(i, j, ch) = median(window(:));
            end
        end
    end
end

function [magnitude, direction] = sobel_edge_detection(img)
    % Sobel edge detection operator
    %
    % Inputs:
    %   img - input grayscale image
    %
    % Outputs:
    %   magnitude - edge magnitude
    %   direction - edge direction in radians
    
    % Convert to grayscale if needed
    if size(img, 3) > 1
        img = rgb2gray_custom(img);
    end
    
    % Sobel kernels
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    
    % Compute gradients
    grad_x = conv2(img, sobel_x, 'same');
    grad_y = conv2(img, sobel_y, 'same');
    
    % Compute magnitude and direction
    magnitude = sqrt(grad_x.^2 + grad_y.^2);
    direction = atan2(grad_y, grad_x);
end

function edges = canny_edge_detection(img, varargin)
    % Canny edge detection algorithm
    %
    % Inputs:
    %   img - input grayscale image
    %   varargin - optional parameters:
    %     'sigma' - Gaussian blur sigma (default: 1)
    %     'low_threshold' - low threshold for hysteresis (default: 0.1)
    %     'high_threshold' - high threshold for hysteresis (default: 0.2)
    %
    % Output:
    %   edges - binary edge map
    
    % Default parameters
    sigma = 1;
    low_threshold = 0.1;
    high_threshold = 0.2;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'sigma'
                sigma = varargin{i+1};
            case 'low_threshold'
                low_threshold = varargin{i+1};
            case 'high_threshold'
                high_threshold = varargin{i+1};
        end
    end
    
    % Convert to grayscale if needed
    if size(img, 3) > 1
        img = rgb2gray_custom(img);
    end
    
    % Step 1: Gaussian smoothing
    smoothed = apply_gaussian_filter(img, sigma);
    
    % Step 2: Gradient calculation
    [magnitude, direction] = sobel_edge_detection(smoothed);
    
    % Normalize magnitude
    magnitude = magnitude / max(magnitude(:));
    
    % Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, direction);
    
    % Step 4: Double thresholding and hysteresis
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold);
end

function suppressed = non_maximum_suppression(magnitude, direction)
    % Non-maximum suppression for edge thinning
    
    [h, w] = size(magnitude);
    suppressed = zeros(h, w);
    
    % Convert direction to degrees and normalize to [0, 180)
    angle = mod(direction * 180 / pi + 180, 180);
    
    for i = 2:h-1
        for j = 2:w-1
            % Determine gradient direction
            if (angle(i,j) >= 0 && angle(i,j) < 22.5) || (angle(i,j) >= 157.5 && angle(i,j) <= 180)
                % Horizontal edge
                neighbors = [magnitude(i, j-1), magnitude(i, j+1)];
            elseif angle(i,j) >= 22.5 && angle(i,j) < 67.5
                % Diagonal edge (\)
                neighbors = [magnitude(i-1, j+1), magnitude(i+1, j-1)];
            elseif angle(i,j) >= 67.5 && angle(i,j) < 112.5
                % Vertical edge
                neighbors = [magnitude(i-1, j), magnitude(i+1, j)];
            else
                % Diagonal edge (/)
                neighbors = [magnitude(i-1, j-1), magnitude(i+1, j+1)];
            end
            
            % Suppress if not maximum
            if magnitude(i,j) >= max(neighbors)
                suppressed(i,j) = magnitude(i,j);
            end
        end
    end
end

function edges = hysteresis_thresholding(img, low_thresh, high_thresh)
    % Hysteresis thresholding for Canny edge detection
    
    [h, w] = size(img);
    edges = zeros(h, w);
    
    % Strong edges (above high threshold)
    strong_edges = img > high_thresh;
    
    % Weak edges (between thresholds)
    weak_edges = (img >= low_thresh) & (img <= high_thresh);
    
    % Start with strong edges
    edges = strong_edges;
    
    % Connect weak edges to strong edges
    changed = true;
    while changed
        changed = false;
        old_edges = edges;
        
        for i = 2:h-1
            for j = 2:w-1
                if weak_edges(i,j) && ~edges(i,j)
                    % Check 8-connected neighborhood
                    neighborhood = edges(i-1:i+1, j-1:j+1);
                    if any(neighborhood(:))
                        edges(i,j) = 1;
                        changed = true;
                    end
                end
            end
        end
    end
end

function filtered_img = unsharp_masking(img, varargin)
    % Unsharp masking for image sharpening
    %
    % Inputs:
    %   img - input image
    %   varargin - optional parameters:
    %     'sigma' - Gaussian blur sigma (default: 1)
    %     'strength' - sharpening strength (default: 1.5)
    %
    % Output:
    %   filtered_img - sharpened image
    
    % Default parameters
    sigma = 1;
    strength = 1.5;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'sigma'
                sigma = varargin{i+1};
            case 'strength'
                strength = varargin{i+1};
        end
    end
    
    % Create unsharp mask
    blurred = apply_gaussian_filter(img, sigma);
    mask = img - blurred;
    
    % Apply sharpening
    filtered_img = img + strength * mask;
    
    % Clamp values
    if max(img(:)) <= 1
        filtered_img = max(0, min(1, filtered_img));
    else
        filtered_img = max(0, min(255, filtered_img));
    end
end

function demo_basic_filters()
    % Demonstrate all basic filtering operations
    
    fprintf('\n--- Basic Filters Demonstration ---\n');
    
    % Load test image
    img = load_image('', 'grayscale', true, 'normalize', true);
    
    % Add noise for demonstration
    noisy_img = img + 0.1 * randn(size(img));
    noisy_img = max(0, min(1, noisy_img));
    
    % Apply different filters
    gaussian_filtered = apply_gaussian_filter(noisy_img, 2);
    median_filtered = apply_median_filter(noisy_img, 5);
    [sobel_mag, sobel_dir] = sobel_edge_detection(img);
    canny_edges = canny_edge_detection(img, 'sigma', 1, 'low_threshold', 0.1, 'high_threshold', 0.2);
    unsharp_filtered = unsharp_masking(img, 'sigma', 1, 'strength', 2);
    
    % Display results
    figure('Position', [50, 50, 1400, 1000]);
    
    subplot(3, 3, 1);
    imshow(img, []);
    title('Original Image');
    
    subplot(3, 3, 2);
    imshow(noisy_img, []);
    title('Noisy Image');
    
    subplot(3, 3, 3);
    imshow(gaussian_filtered, []);
    title('Gaussian Filtered');
    
    subplot(3, 3, 4);
    imshow(median_filtered, []);
    title('Median Filtered');
    
    subplot(3, 3, 5);
    imshow(sobel_mag, []);
    title('Sobel Edge Magnitude');
    
    subplot(3, 3, 6);
    imshow(sobel_dir, []);
    title('Sobel Edge Direction');
    colormap(gca, hsv);
    
    subplot(3, 3, 7);
    imshow(canny_edges, []);
    title('Canny Edges');
    
    subplot(3, 3, 8);
    imshow(unsharp_filtered, []);
    title('Unsharp Masking');
    
    subplot(3, 3, 9);
    % Comparison plot
    plot_line_profile(img, noisy_img, gaussian_filtered, median_filtered);
    title('Filter Comparison (Line Profile)');
    
    sgtitle('Basic Image Filters Demonstration');
    
    fprintf('Basic filters demonstration complete.\n');
end

function plot_line_profile(original, noisy, gaussian, median)
    % Plot line profiles for filter comparison
    
    [h, w] = size(original);
    row = round(h/2); % Middle row
    
    x = 1:w;
    plot(x, original(row, :), 'g', 'LineWidth', 2); hold on;
    plot(x, noisy(row, :), 'r', 'LineWidth', 1);
    plot(x, gaussian(row, :), 'b', 'LineWidth', 1.5);
    plot(x, median(row, :), 'm', 'LineWidth', 1.5);
    
    legend('Original', 'Noisy', 'Gaussian', 'Median', 'Location', 'best');
    xlabel('Pixel Position');
    ylabel('Intensity');
    grid on;
end

function compare_edge_detectors(img)
    % Compare different edge detection methods
    
    % Convert to grayscale if needed
    if size(img, 3) > 1
        img = rgb2gray_custom(img);
    end
    
    % Apply different edge detection methods
    [sobel_mag, ~] = sobel_edge_detection(img);
    canny_edges = canny_edge_detection(img, 'sigma', 1);
    
    % Simple gradient-based edge detection
    [gx, gy] = gradient(img);
    gradient_mag = sqrt(gx.^2 + gy.^2);
    
    % Laplacian edge detection
    laplacian_kernel = [0 1 0; 1 -4 1; 0 1 0];
    laplacian_edges = abs(conv2(img, laplacian_kernel, 'same'));
    
    % Display comparison
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    imshow(img, []);
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(sobel_mag, []);
    title('Sobel Edges');
    
    subplot(2, 3, 3);
    imshow(gradient_mag, []);
    title('Gradient Magnitude');
    
    subplot(2, 3, 4);
    imshow(laplacian_edges, []);
    title('Laplacian Edges');
    
    subplot(2, 3, 5);
    imshow(canny_edges, []);
    title('Canny Edges');
    
    subplot(2, 3, 6);
    % Edge strength histogram
    edges_data = {sobel_mag(:), gradient_mag(:), laplacian_edges(:)};
    edge_names = {'Sobel', 'Gradient', 'Laplacian'};
    colors = {'r', 'g', 'b'};
    
    for i = 1:length(edges_data)
        [counts, bins] = hist(edges_data{i}, 50);
        counts = counts / sum(counts); % Normalize
        plot(bins, counts, colors{i}, 'LineWidth', 1.5); hold on;
    end
    
    legend(edge_names, 'Location', 'best');
    title('Edge Strength Distributions');
    xlabel('Edge Strength');
    ylabel('Normalized Frequency');
    grid on;
    
    sgtitle('Edge Detection Methods Comparison');
    
    fprintf('Edge detection comparison complete.\n');
end

function filtered_img = bilateral_filter(img, sigma_spatial, sigma_intensity, varargin)
    % Bilateral filter for edge-preserving smoothing
    %
    % Inputs:
    %   img - input image
    %   sigma_spatial - spatial standard deviation
    %   sigma_intensity - intensity standard deviation
    %   varargin - optional parameters:
    %     'kernel_size' - filter window size
    %
    % Output:
    %   filtered_img - bilateral filtered image
    
    % Default parameters
    kernel_size = 2 * ceil(3 * sigma_spatial) + 1;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'kernel_size'
                kernel_size = varargin{i+1};
        end
    end
    
    % Ensure odd kernel size
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
    end
    
    [h, w, c] = size(img);
    filtered_img = zeros(size(img));
    pad_size = floor(kernel_size / 2);
    
    % Create spatial weight matrix
    [x, y] = meshgrid(-pad_size:pad_size);
    spatial_weights = exp(-(x.^2 + y.^2) / (2 * sigma_spatial^2));
    
    % Process each channel
    for ch = 1:c
        channel = img(:,:,ch);
        padded_channel = padarray(channel, [pad_size, pad_size], 'replicate');
        
        for i = 1:h
            for j = 1:w
                % Extract neighborhood
                neighborhood = padded_channel(i:i+kernel_size-1, j:j+kernel_size-1);
                center_value = channel(i, j);
                
                % Compute intensity weights
                intensity_diff = neighborhood - center_value;
                intensity_weights = exp(-intensity_diff.^2 / (2 * sigma_intensity^2));
                
                % Combine spatial and intensity weights
                combined_weights = spatial_weights .* intensity_weights;
                combined_weights = combined_weights / sum(combined_weights(:));
                
                % Apply bilateral filter
                filtered_img(i, j, ch) = sum(sum(neighborhood .* combined_weights));
            end
        end
    end
end

function demo_advanced_filters()
    % Demonstrate advanced filtering techniques
    
    fprintf('\n--- Advanced Filters Demonstration ---\n');
    
    % Create test image with noise
    img = load_image('', 'normalize', true);
    if size(img, 3) > 1
        img = rgb2gray_custom(img);
    end
    
    % Add different types of noise
    gaussian_noise_img = img + 0.1 * randn(size(img));
    gaussian_noise_img = max(0, min(1, gaussian_noise_img));
    
    % Salt and pepper noise
    salt_pepper_img = img;
    noise_mask = rand(size(img));
    salt_pepper_img(noise_mask < 0.05) = 0; % Salt
    salt_pepper_img(noise_mask > 0.95) = 1; % Pepper
    
    % Apply different filters
    gaussian_result = apply_gaussian_filter(gaussian_noise_img, 1.5);
    median_result = apply_median_filter(salt_pepper_img, 5);
    bilateral_result = bilateral_filter(gaussian_noise_img, 2, 0.1);
    
    % Display results
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 4, 1);
    imshow(img, []);
    title('Original');
    
    subplot(2, 4, 2);
    imshow(gaussian_noise_img, []);
    title('Gaussian Noise');
    
    subplot(2, 4, 3);
    imshow(salt_pepper_img, []);
    title('Salt & Pepper Noise');
    
    subplot(2, 4, 4);
    imshow(gaussian_result, []);
    title('Gaussian Filter');
    
    subplot(2, 4, 5);
    imshow(median_result, []);
    title('Median Filter');
    
    subplot(2, 4, 6);
    imshow(bilateral_result, []);
    title('Bilateral Filter');
    
    % Edge detection comparison
    subplot(2, 4, 7);
    compare_edge_detectors(img);
    
    subplot(2, 4, 8);
    % Filter performance metrics
    psnr_gaussian = calculate_psnr(img, gaussian_result);
    psnr_median = calculate_psnr(img, median_result);
    psnr_bilateral = calculate_psnr(img, bilateral_result);
    
    methods = {'Gaussian', 'Median', 'Bilateral'};
    psnr_values = [psnr_gaussian, psnr_median, psnr_bilateral];
    
    bar(psnr_values);
    set(gca, 'XTickLabel', methods);
    title('Filter PSNR Comparison');
    ylabel('PSNR (dB)');
    grid on;
    
    sgtitle('Advanced Image Filtering Techniques');
    
    fprintf('Advanced filters demonstration complete.\n');
    fprintf('PSNR Results:\n');
    fprintf('  Gaussian: %.2f dB\n', psnr_gaussian);
    fprintf('  Median: %.2f dB\n', psnr_median);
    fprintf('  Bilateral: %.2f dB\n', psnr_bilateral);
end

function psnr_val = calculate_psnr(original, processed)
    % Calculate Peak Signal-to-Noise Ratio
    
    mse = mean((original(:) - processed(:)).^2);
    if mse == 0
        psnr_val = inf;
    else
        max_val = max(original(:));
        psnr_val = 20 * log10(max_val / sqrt(mse));
    end
end