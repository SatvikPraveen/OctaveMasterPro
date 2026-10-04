% Location: mini_projects/parallel_image_batch_processing/private/create_test_image_for_operations.m
% (split out of image_operations.m so it can be called from other files)

function test_img = create_test_image_for_operations()
    # Create comprehensive test image
    
    img_size = 300;
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # Create RGB test image
    test_img = zeros(img_size, img_size, 3);
    
    # Red channel - circles
    center1 = [100, 100];
    center2 = [200, 200];
    radius1 = 40;
    radius2 = 30;
    
    circle1 = (x - center1(1)).^2 + (y - center1(2)).^2 <= radius1^2;
    circle2 = (x - center2(1)).^2 + (y - center2(2)).^2 <= radius2^2;
    
    test_img(:,:,1) = double(circle1) * 0.8 + double(circle2) * 0.6;
    
    # Green channel - stripes
    test_img(:,:,2) = 0.5 + 0.3 * sin(2*pi*x/50);
    
    # Blue channel - gradient
    test_img(:,:,3) = (x + y) / (2 * img_size);
    
    # Add some noise
    test_img = test_img + 0.05 * randn(size(test_img));
    
    # Clamp to valid range
    test_img = max(0, min(1, test_img));
end
