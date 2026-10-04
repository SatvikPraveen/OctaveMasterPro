% Location: mini_projects/image_processing_basics/private/sobel_edge_detection.m
% (split out of basic_filters.m so it can be called from other files)

function [magnitude, direction] = sobel_edge_detection(img)
    % Sobel edge detection operator
    %
    % Inputs:
    %   img - input grayscale image
    %
    % Outputs:
    %   magnitude - edge magnitude
    %   direction - edge direction in radians
    
    % Convert to grayscale if needed
    if size(img, 3) > 1
        img = rgb2gray_custom(img);
    end
    
    % Sobel kernels
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    
    % Compute gradients
    grad_x = conv2(img, sobel_x, 'same');
    grad_y = conv2(img, sobel_y, 'same');
    
    % Compute magnitude and direction
    magnitude = sqrt(grad_x.^2 + grad_y.^2);
    direction = atan2(grad_y, grad_x);
end
