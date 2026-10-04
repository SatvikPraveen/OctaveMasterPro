% Location: mini_projects/image_processing_basics/private/demo_advanced_morphology.m
% (split out of morphology_ops.m so it can be called from other files)

function demo_advanced_morphology()
    % Demonstrate advanced morphological operations
    
    fprintf('\n--- Advanced Morphology Demonstration ---\n');
    
    % Create complex test image
    img = create_complex_binary_image();
    
    % Apply advanced operations
    se = create_structuring_element('disk', 2);
    
    boundary = extract_boundary(img, se);
    skeleton = morphological_skeletonization(img);
    gradient = morphological_gradient(img, se);
    tophat = top_hat_transform(img, create_structuring_element('disk', 5));
    
    % Display results
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    imshow(img, []);
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(boundary, []);
    title('Boundary Extraction');
    
    subplot(2, 3, 3);
    imshow(skeleton, []);
    title('Morphological Skeleton');
    
    subplot(2, 3, 4);
    imshow(gradient, []);
    title('Morphological Gradient');
    
    subplot(2, 3, 5);
    imshow(tophat, []);
    title('Top-hat Transform');
    
    subplot(2, 3, 6);
    % Overlay boundary on original
    overlay = cat(3, img, img + boundary, img);
    imshow(overlay, []);
    title('Boundary Overlay');
    
    sgtitle('Advanced Morphological Operations');
    
    fprintf('Advanced morphology demonstration complete.\n');
end
