% Location: mini_projects/image_processing_basics/private/create_binary_test_image.m
% (split out of morphology_ops.m so it can be called from other files)

function binary_img = create_binary_test_image()
    % Create binary test image with various shapes
    
    img = zeros(200, 200);
    
    % Add circles
    [x, y] = meshgrid(1:200, 1:200);
    img = img + double((x-50).^2 + (y-50).^2 < 20^2);
    img = img + double((x-150).^2 + (y-50).^2 < 15^2);
    img = img + double((x-50).^2 + (y-150).^2 < 25^2);
    
    % Add rectangles
    img(100:130, 100:140) = 1;
    img(160:180, 120:180) = 1;
    
    % Add line structures
    img(80:120, 75:77) = 1;
    img(140:142, 80:120) = 1;
    
    binary_img = img > 0;
end
