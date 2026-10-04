% Location: mini_projects/image_processing_basics/private/demo_histogram_processing.m
% (split out of histogram_analysis.m so it can be called from other files)

function demo_histogram_processing()
    % Comprehensive histogram processing demonstration
    
    fprintf('\n--- Histogram Processing Demonstration ---\n');
    
    % Create test images with different contrast characteristics
    low_contrast = create_low_contrast_image();
    high_contrast = create_high_contrast_image();
    dark_image = create_dark_image();
    
    % Apply histogram processing
    eq_low = histogram_equalization(low_contrast);
    eq_high = histogram_equalization(high_contrast);
    eq_dark = histogram_equalization(dark_image);
    
    % Adaptive equalization
    adaptive_low = adaptive_histogram_equalization(low_contrast, 'tile_size', [4, 4]);
    adaptive_dark = adaptive_histogram_equalization(dark_image, 'tile_size', [6, 6]);
    
    % Display results
    figure('Position', [50, 50, 1400, 1000]);
    
    % Original images
    subplot(4, 4, 1); imshow(low_contrast, []); title('Low Contrast Original');
    subplot(4, 4, 2); imshow(high_contrast, []); title('High Contrast Original');
    subplot(4, 4, 3); imshow(dark_image, []); title('Dark Image Original');
    subplot(4, 4, 4); plot_histogram_comparison(low_contrast, 'Original Low Contrast');
    
    % Histogram equalized
    subplot(4, 4, 5); imshow(eq_low, []); title('HE: Low Contrast');
    subplot(4, 4, 6); imshow(eq_high, []); title('HE: High Contrast');
    subplot(4, 4, 7); imshow(eq_dark, []); title('HE: Dark Image');
    subplot(4, 4, 8); plot_histogram_comparison(eq_low, 'Equalized Low Contrast');
    
    % Adaptive equalized
    subplot(4, 4, 9); imshow(adaptive_low, []); title('CLAHE: Low Contrast');
    subplot(4, 4, 10); imshow(adaptive_dark, []); title('CLAHE: Dark Image');
    
    % Histogram matching demo
    matched = histogram_matching(dark_image, high_contrast);
    subplot(4, 4, 11); imshow(matched, []); title('Histogram Matched');
    subplot(4, 4, 12); plot_histogram_comparison(matched, 'Matched Histogram');
    
    % Statistics comparison
    subplot(4, 4, [13, 14, 15, 16]);
    compare_image_statistics({low_contrast, eq_low, adaptive_low}, ...
                           {'Original', 'Equalized', 'CLAHE'});
    
    sgtitle('Comprehensive Histogram Processing');
    
    fprintf('Histogram processing demonstration complete.\n');
end
