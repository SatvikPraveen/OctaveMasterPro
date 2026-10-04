% Location: mini_projects/parallel_image_batch_processing/private/generate_test_images.m
% (split out of batch_processor.m so it can be called from other files)

function test_files = generate_test_images(output_dir, num_images)
    % Generate synthetic test images
    
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    test_files = cell(num_images, 1);
    
    fprintf('Generating %d test images...\n', num_images);
    
    for i = 1:num_images
        # Create synthetic image
        img_size = 200 + round(100 * rand()); # Random size 200-300
        [x, y] = meshgrid(1:img_size, 1:img_size);
        
        # Random pattern type
        pattern_type = mod(i, 4) + 1;
        
        switch pattern_type
            case 1 # Circles
                img = create_circle_pattern(x, y);
            case 2 # Stripes
                img = create_stripe_pattern(x, y, img_size);
            case 3 # Noise
                img = create_noise_pattern(img_size);
            case 4 # Gradient
                img = create_gradient_pattern(x, y);
        end
        
        # Add some noise
        img = img + 0.1 * randn(size(img));
        img = max(0, min(255, img));
        
        # Save test image
        filename = sprintf('test_image_%03d.png', i);
        filepath = fullfile(output_dir, filename);
        
        try
            imwrite(uint8(img), filepath);
            test_files{i} = filepath;
        catch
            fprintf('Warning: Could not save test image %d\n', i);
        end
    end
    
    # Remove empty entries
    test_files = test_files(~cellfun(@isempty, test_files));
    
    fprintf('Generated %d test images in %s\n', length(test_files), output_dir);
end
