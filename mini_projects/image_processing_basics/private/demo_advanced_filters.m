% Location: mini_projects/image_processing_basics/private/demo_advanced_filters.m
% (split out of basic_filters.m so it can be called from other files)

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
