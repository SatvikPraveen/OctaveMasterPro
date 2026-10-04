% Location: mini_projects/image_processing_basics/private/noise_removal_demo.m
% (split out of morphology_ops.m so it can be called from other files)

function cleaned_img = noise_removal_demo(img)
    % Demonstrate morphological noise removal
    
    fprintf('\n--- Morphological Noise Removal ---\n');
    
    % Convert to binary if needed
    if max(img(:)) > 1
        binary_img = img > 128;
    else
        binary_img = img > 0.5;
    end
    
    % Add salt and pepper noise
    noisy_img = binary_img;
    noise_density = 0.05;
    
    % Salt noise (random white pixels)
    salt_locations = rand(size(binary_img)) < noise_density/2;
    noisy_img(salt_locations) = 1;
    
    % Pepper noise (random black pixels)
    pepper_locations = rand(size(binary_img)) < noise_density/2;
    noisy_img(pepper_locations) = 0;
    
    % Apply morphological cleaning
    se_small = create_structuring_element('disk', 1);
    se_medium = create_structuring_element('disk', 2);
    
    % Method 1: Opening followed by closing
    method1 = morphological_opening(noisy_img, se_small);
    method1 = morphological_closing(method1, se_small);
    
    % Method 2: Median-like morphological filtering
    method2 = morphological_closing(noisy_img, se_small);
    method2 = morphological_opening(method2, se_small);
    
    % Method 3: Multiple iterations with small SE
    method3 = noisy_img;
    for iter = 1:2
        method3 = morphological_opening(method3, se_small);
        method3 = morphological_closing(method3, se_small);
    end
    
    % Display results
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    imshow(binary_img, []);
    title('Original Binary Image');
    
    subplot(2, 3, 2);
    imshow(noisy_img, []);
    title('Salt & Pepper Noise Added');
    
    subplot(2, 3, 3);
    imshow(method1, []);
    title('Method 1: Open→Close');
    
    subplot(2, 3, 4);
    imshow(method2, []);
    title('Method 2: Close→Open');
    
    subplot(2, 3, 5);
    imshow(method3, []);
    title('Method 3: Iterative');
    
    % Performance comparison
    subplot(2, 3, 6);
    error1 = sum(sum(abs(binary_img - method1)));
    error2 = sum(sum(abs(binary_img - method2)));
    error3 = sum(sum(abs(binary_img - method3)));
    
    methods = {'Open→Close', 'Close→Open', 'Iterative'};
    errors = [error1, error2, error3];
    
    bar(errors);
    set(gca, 'XTickLabel', methods);
    title('Reconstruction Error');
    ylabel('Total Pixel Differences');
    grid on;
    
    sgtitle('Morphological Noise Removal Comparison');
    
    cleaned_img = method1; % Return best result
    
    fprintf('Noise removal demonstration complete.\n');
    fprintf('Reconstruction errors: %.0f, %.0f, %.0f\n', error1, error2, error3);
end
