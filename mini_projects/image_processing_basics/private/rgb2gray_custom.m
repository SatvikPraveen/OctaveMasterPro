% Location: mini_projects/image_processing_basics/private/rgb2gray_custom.m
% (split out of image_loader.m so it can be called from other files)

function gray_img = rgb2gray_custom(rgb_img)
    % Convert RGB to grayscale using standard weights
    
    if size(rgb_img, 3) ~= 3
        gray_img = rgb_img;
        return;
    end
    
    % Standard RGB to grayscale conversion weights
    weights = [0.299, 0.587, 0.114];
    gray_img = rgb_img(:,:,1) * weights(1) + ...
               rgb_img(:,:,2) * weights(2) + ...
               rgb_img(:,:,3) * weights(3);
end
