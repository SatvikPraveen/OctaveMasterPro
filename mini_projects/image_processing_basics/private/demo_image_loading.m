% Location: mini_projects/image_processing_basics/private/demo_image_loading.m
% (split out of image_loader.m so it can be called from other files)

function demo_image_loading()
    % Demonstrate image loading capabilities
    
    fprintf('\n--- Image Loading Demonstration ---\n');
    
    % Create various test images
    test_images = cell(4, 1);
    test_images{1} = create_test_image('circles');
    test_images{2} = create_test_image('stripes');
    test_images{3} = create_test_image('mixed');
    test_images{4} = create_test_image('gradient');
    
    titles = {'Circles', 'Stripes', 'Mixed Pattern', 'Gradient'};
    
    figure('Position', [100, 100, 1200, 800]);
    
    for i = 1:4
        subplot(2, 2, i);
        imshow(test_images{i}, []);
        title(titles{i});
        
        % Show image info
        get_image_info(double(test_images{i}));
    end
    
    sgtitle('Test Image Gallery');
    
    fprintf('Image loading demonstration complete.\n');
end
