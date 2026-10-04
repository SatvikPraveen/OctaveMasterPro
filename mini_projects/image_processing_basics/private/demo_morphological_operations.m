% Location: mini_projects/image_processing_basics/private/demo_morphological_operations.m
% (split out of morphology_ops.m so it can be called from other files)

function demo_morphological_operations()
    % Demonstrate all morphological operations
    
    fprintf('\n--- Morphological Operations Demonstration ---\n');
    
    % Create binary test image
    img = create_binary_test_image();
    
    % Create different structuring elements
    se_disk = create_structuring_element('disk', 3);
    se_square = create_structuring_element('square', 5);
    se_cross = create_structuring_element('cross', 2);
    
    % Apply operations
    eroded = morphological_erosion(img, se_disk);
    dilated = morphological_dilation(img, se_disk);
    opened = morphological_opening(img, se_square);
    closed = morphological_closing(img, se_square);
    gradient = morphological_gradient(img, se_cross);
    tophat = top_hat_transform(img, se_disk);
    blackhat = black_hat_transform(img, se_disk);
    
    % Display results
    figure('Position', [50, 50, 1400, 1000]);
    
    subplot(3, 3, 1);
    imshow(img, []);
    title('Original Binary Image');
    
    subplot(3, 3, 2);
    imshow(eroded, []);
    title('Erosion (Disk r=3)');
    
    subplot(3, 3, 3);
    imshow(dilated, []);
    title('Dilation (Disk r=3)');
    
    subplot(3, 3, 4);
    imshow(opened, []);
    title('Opening (Square 5x5)');
    
    subplot(3, 3, 5);
    imshow(closed, []);
    title('Closing (Square 5x5)');
    
    subplot(3, 3, 6);
    imshow(gradient, []);
    title('Morphological Gradient');
    
    subplot(3, 3, 7);
    imshow(tophat, []);
    title('Top-hat Transform');
    
    subplot(3, 3, 8);
    imshow(blackhat, []);
    title('Black-hat Transform');
    
    subplot(3, 3, 9);
    % Show structuring elements
    subplot(3, 3, 9);
    se_display = zeros(15, 45);
    se_display(6:10, 2:6) = se_square;
    se_display(6:12, 15:21) = se_disk;
    se_display(6:10, 35:39) = se_cross;
    imshow(se_display, []);
    title('Structuring Elements');
    
    sgtitle('Mathematical Morphology Operations');
    
    fprintf('Morphological operations demonstration complete.\n');
end
