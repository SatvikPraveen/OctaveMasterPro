% Location: mini_projects/parallel_image_batch_processing/private/demo_image_operations.m
% (split out of image_operations.m so it can be called from other files)

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
