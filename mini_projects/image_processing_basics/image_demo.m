% Location: mini_projects/image_processing_basics/image_demo.m
% Main Image Processing Demonstration Script

function image_demo()
    % Complete image processing demonstration
    
    clear; clc; close all;
    
    fprintf('====================================================\n');
    fprintf('      IMAGE PROCESSING BASICS DEMONSTRATION        \n');
    fprintf('====================================================\n\n');
    
    try
        while true
            fprintf('\nSelect a demonstration:\n');
            fprintf('1. Image Loading & Preprocessing\n');
            fprintf('2. Basic Filters (Gaussian, Median, Edge)\n');
            fprintf('3. Morphological Operations\n');
            fprintf('4. Histogram Analysis & Enhancement\n');
            fprintf('5. Complete Processing Pipeline\n');
            fprintf('6. Interactive Image Lab\n');
            fprintf('0. Exit\n');
            
            choice = input('Enter your choice (0-6): ');
            
            switch choice
                case 0
                    fprintf('\nExiting Image Processing Demo. Goodbye!\n');
                    break;
                case 1
                    image_loading_demo();
                case 2
                    basic_filters_demo();
                case 3
                    morphology_demo();
                case 4
                    histogram_demo();
                case 5
                    complete_pipeline_demo();
                case 6
                    interactive_image_lab();
                otherwise
                    fprintf('Invalid choice. Please select 0-6.\n');
            end
            
            if choice ~= 0
                input('\nPress Enter to continue...');
            end
        end
        
    catch err
        fprintf('Error in image_demo: %s\n', err.message);
        fprintf('Make sure all required functions are available.\n');
    end
end

function image_loading_demo()
    fprintf('\n--- Image Loading & Preprocessing Demo ---\n');
    demo_image_loading();
end

function basic_filters_demo()
    fprintf('\n--- Basic Filters Demo ---\n');
    demo_basic_filters();
    
    % Additional edge detection comparison
    fprintf('\nRunning edge detection comparison...\n');
    test_img = load_image('', 'grayscale', true, 'normalize', true);
    compare_edge_detectors(test_img);
    
    % Advanced filters
    fprintf('\nRunning advanced filters demo...\n');
    demo_advanced_filters();
end

function morphology_demo()
    fprintf('\n--- Morphological Operations Demo ---\n');
    demo_morphological_operations();
    
    % Advanced morphology
    fprintf('\nRunning advanced morphology demo...\n');
    demo_advanced_morphology();
    
    % Noise removal
    fprintf('\nRunning noise removal demo...\n');
    test_img = create_binary_test_image();
    noise_removal_demo(test_img);
end

function histogram_demo()
    fprintf('\n--- Histogram Analysis Demo ---\n');
    demo_histogram_processing();
end

function complete_pipeline_demo()
    % Demonstrate complete image processing pipeline
    
    fprintf('\n--- Complete Image Processing Pipeline ---\n');
    fprintf('Processing a realistic image enhancement workflow...\n');
    
    % Step 1: Load and preprocess
    fprintf('1. Loading and preprocessing image...\n');
    original = load_image('', 'normalize', true);
    if size(original, 3) > 1
        gray_img = rgb2gray_custom(original);
    else
        gray_img = original;
    end
    
    % Step 2: Noise reduction
    fprintf('2. Applying noise reduction...\n');
    denoised = apply_gaussian_filter(gray_img, 0.8);
    
    % Step 3: Contrast enhancement
    fprintf('3. Enhancing contrast...\n');
    enhanced = adaptive_histogram_equalization(denoised, 'tile_size', [8, 8]);
    
    % Step 4: Edge detection
    fprintf('4. Detecting edges...\n');
    edges = canny_edge_detection(enhanced, 'sigma', 1, 'low_threshold', 0.1, 'high_threshold', 0.25);
    
    % Step 5: Morphological processing
    fprintf('5. Morphological refinement...\n');
    se = create_structuring_element('disk', 1);
    cleaned_edges = morphological_closing(edges, se);
    cleaned_edges = morphological_opening(cleaned_edges, se);
    
    % Step 6: Final enhancement
    fprintf('6. Final sharpening...\n');
    final_result = unsharp_masking(enhanced, 'sigma', 1, 'strength', 1.5);
    
    % Display complete pipeline
    figure('Position', [50, 50, 1400, 1000]);
    
    subplot(3, 3, 1); imshow(gray_img, []); title('1. Original');
    subplot(3, 3, 2); imshow(denoised, []); title('2. Denoised');
    subplot(3, 3, 3); imshow(enhanced, []); title('3. Contrast Enhanced');
    subplot(3, 3, 4); imshow(edges, []); title('4. Edge Detection');
    subplot(3, 3, 5); imshow(cleaned_edges, []); title('5. Cleaned Edges');
    subplot(3, 3, 6); imshow(final_result, []); title('6. Final Result');
    
    % Histogram progression
    subplot(3, 3, 7);
    [h1, b1] = compute_histogram(gray_img, 64);
    [h2, b2] = compute_histogram(enhanced, 64);
    [h3, b3] = compute_histogram(final_result, 64);
    
    plot(b1, h1, 'r', 'LineWidth', 1.5); hold on;
    plot(b2, h2, 'g', 'LineWidth', 1.5);
    plot(b3, h3, 'b', 'LineWidth', 1.5);
    legend('Original', 'Enhanced', 'Final', 'Location', 'best');
    title('Histogram Evolution');
    xlabel('Intensity'); ylabel('Frequency'); grid on;
    
    % Quality metrics
    subplot(3, 3, 8);
    stats_orig = analyze_histogram_statistics(gray_img);
    stats_final = analyze_histogram_statistics(final_result);
    
    metrics = {'Entropy', 'Contrast', 'Std'};
    orig_vals = [stats_orig.entropy, stats_orig.rms_contrast, stats_orig.std];
    final_vals = [stats_final.entropy, stats_final.rms_contrast, stats_final.std];
    
    x = 1:length(metrics);
    bar(x-0.2, orig_vals/max([orig_vals, final_vals]), 0.4); hold on;
    bar(x+0.2, final_vals/max([orig_vals, final_vals]), 0.4);
    
    set(gca, 'XTickLabel', metrics);
    legend('Original', 'Processed', 'Location', 'best');
    title('Quality Metrics');
    ylabel('Normalized Values'); grid on;
    
    % Before/after comparison
    subplot(3, 3, 9);
    comparison = [gray_img, final_result];
    imshow(comparison, []);
    title('Before | After');
    
    sgtitle('Complete Image Processing Pipeline');
    
    fprintf('7. Pipeline complete! Quality improvements:\n');
    fprintf('   Entropy: %.3f → %.3f\n', stats_orig.entropy, stats_final.entropy);
    fprintf('   Contrast: %.2f → %.2f\n', stats_orig.rms_contrast, stats_final.rms_contrast);
    fprintf('   Std Dev: %.2f → %.2f\n', stats_orig.std, stats_final.std);
end

function interactive_image_lab()
    % Interactive image processing laboratory
    
    fprintf('\n--- Interactive Image Processing Lab ---\n');
    
    while true
        fprintf('\nInteractive Lab Menu:\n');
        fprintf('1. Custom filter parameters\n');
        fprintf('2. Morphology playground\n');
        fprintf('3. Histogram tools\n');
        fprintf('4. Edge detection tuning\n');
        fprintf('0. Return to main menu\n');
        
        choice = input('Enter choice: ');
        
        switch choice
            case 0
                break;
            case 1
                custom_filter_demo();
            case 2
                morphology_playground();
            case 3
                histogram_tools();
            case 4
                edge_detection_tuning();
            otherwise
                fprintf('Invalid choice.\n');
        end
    end
end

function custom_filter_demo()
    fprintf('\n--- Custom Filter Parameters ---\n');
    
    img = load_image('', 'grayscale', true, 'normalize', true);
    
    sigma = input('Enter Gaussian sigma [2]: ');
    if isempty(sigma), sigma = 2; end
    
    kernel_size = input('Enter median filter size [5]: ');
    if isempty(kernel_size), kernel_size = 5; end
    
    gaussian_result = apply_gaussian_filter(img, sigma);
    median_result = apply_median_filter(img, kernel_size);
    
    figure;
    subplot(1, 3, 1); imshow(img, []); title('Original');
    subplot(1, 3, 2); imshow(gaussian_result, []); title(sprintf('Gaussian σ=%.1f', sigma));
    subplot(1, 3, 3); imshow(median_result, []); title(sprintf('Median %dx%d', kernel_size, kernel_size));
    
    fprintf('Custom filtering complete.\n');
end

function morphology_playground()
    fprintf('\n--- Morphology Playground ---\n');
    
    img = create_binary_test_image();
    
    fprintf('Structuring elements: disk, square, cross, line\n');
    shape = input('Enter shape [disk]: ', 's');
    if isempty(shape), shape = 'disk'; end
    
    size_param = input('Enter size [3]: ');
    if isempty(size_param), size_param = 3; end
    
    se = create_structuring_element(shape, size_param);
    
    eroded = morphological_erosion(img, se);
    dilated = morphological_dilation(img, se);
    opened = morphological_opening(img, se);
    closed = morphological_closing(img, se);
    
    figure;
    subplot(2, 3, 1); imshow(img, []); title('Original');
    subplot(2, 3, 2); imshow(eroded, []); title('Erosion');
    subplot(2, 3, 3); imshow(dilated, []); title('Dilation');
    subplot(2, 3, 4); imshow(opened, []); title('Opening');
    subplot(2, 3, 5); imshow(closed, []); title('Closing');
    subplot(2, 3, 6); imshow(se, []); title('Structuring Element');
    
    fprintf('Morphological operations complete.\n');
end

function histogram_tools()
    fprintf('\n--- Histogram Tools ---\n');
    
    img = load_image('', 'grayscale', true);
    
    fprintf('Enhancement methods: equalize, adaptive, match\n');
    method = input('Enter method [equalize]: ', 's');
    if isempty(method), method = 'equalize'; end
    
    switch method
        case 'equalize'
            result = histogram_equalization(img);
        case 'adaptive'
            result = adaptive_histogram_equalization(img);
        case 'match'
            target = create_high_contrast_image();
            result = histogram_matching(img, target);
    end
    
    figure;
    subplot(2, 2, 1); imshow(img, []); title('Original');
    subplot(2, 2, 2); imshow(result, []); title('Enhanced');
    subplot(2, 2, 3); plot_histogram_comparison(img, 'Original Histogram');
    subplot(2, 2, 4); plot_histogram_comparison(result, 'Enhanced Histogram');
    
    fprintf('Histogram processing complete.\n');
end

function edge_detection_tuning()
    fprintf('\n--- Edge Detection Tuning ---\n');
    
    img = load_image('', 'grayscale', true, 'normalize', true);
    
    sigma = input('Enter Gaussian sigma [1]: ');
    if isempty(sigma), sigma = 1; end
    
    low_thresh = input('Enter low threshold [0.1]: ');
    if isempty(low_thresh), low_thresh = 0.1; end
    
    high_thresh = input('Enter high threshold [0.2]: ');
    if isempty(high_thresh), high_thresh = 0.2; end
    
    edges = canny_edge_detection(img, 'sigma', sigma, 'low_threshold', low_thresh, 'high_threshold', high_thresh);
    [sobel_mag, ~] = sobel_edge_detection(img);
    
    figure;
    subplot(1, 3, 1); imshow(img, []); title('Original');
    subplot(1, 3, 2); imshow(sobel_mag, []); title('Sobel Edges');
    subplot(1, 3, 3); imshow(edges, []); title(sprintf('Canny (σ=%.1f)', sigma));
    
    fprintf('Edge detection tuning complete.\n');
end