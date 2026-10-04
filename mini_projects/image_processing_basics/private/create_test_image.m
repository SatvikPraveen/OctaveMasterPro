% Location: mini_projects/image_processing_basics/private/create_test_image.m
% (split out of image_loader.m so it can be called from other files)

function test_img = create_test_image(type)
    % Create synthetic test images for demonstration
    
    if nargin < 1
        type = 'mixed';
    end
    
    switch lower(type)
        case 'circles'
            % Image with circles of different sizes
            [x, y] = meshgrid(-100:100, -100:100);
            test_img = zeros(201, 201);
            
            % Add circles
            test_img = test_img + double((x.^2 + y.^2) < 30^2) * 255;
            test_img = test_img + double(((x-50).^2 + (y-30).^2) < 20^2) * 180;
            test_img = test_img + double(((x+40).^2 + (y+40).^2) < 15^2) * 120;
            
        case 'stripes'
            % Vertical and horizontal stripes
            test_img = zeros(200, 200);
            test_img(:, 20:40) = 255;
            test_img(:, 80:100) = 200;
            test_img(:, 140:160) = 150;
            test_img(60:80, :) = test_img(60:80, :) + 100;
            test_img = min(test_img, 255);
            
        case 'mixed'
            % Complex test pattern
            [x, y] = meshgrid(1:256, 1:256);
            test_img = zeros(256, 256);
            
            % Geometric shapes
            test_img = test_img + double((x-64).^2 + (y-64).^2 < 30^2) * 200;
            test_img = test_img + double(abs(x-192) < 20 & abs(y-64) < 20) * 180;
            test_img = test_img + double(abs(x-64-y+192) < 10) * 160;
            
            % Add noise and texture
            test_img = test_img + 30 * randn(256, 256);
            test_img = max(0, min(255, test_img));
            
        case 'gradient'
            % Smooth gradient for testing filters
            [x, y] = meshgrid(1:200, 1:200);
            test_img = x + y;
            test_img = 255 * test_img / max(test_img(:));
            
        otherwise
            % Default: simple geometric pattern
            test_img = create_test_image('mixed');
    end
    
    test_img = uint8(test_img);
end
