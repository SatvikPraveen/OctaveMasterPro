% Location: mini_projects/image_processing_basics/private/demo_basic_filters.m
% (split out of basic_filters.m so it can be called from other files)

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
