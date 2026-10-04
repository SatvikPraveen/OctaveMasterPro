% Location: mini_projects/image_processing_basics/private/canny_edge_detection.m
% (split out of basic_filters.m so it can be called from other files)

function edges = canny_edge_detection(img, varargin)
    % Canny edge detection algorithm
    %
    % Inputs:
    %   img - input grayscale image
    %   varargin - optional parameters:
    %     'sigma' - Gaussian blur sigma (default: 1)
    %     'low_threshold' - low threshold for hysteresis (default: 0.1)
    %     'high_threshold' - high threshold for hysteresis (default: 0.2)
    %
    % Output:
    %   edges - binary edge map
    
    % Default parameters
    sigma = 1;
    low_threshold = 0.1;
    high_threshold = 0.2;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'sigma'
                sigma = varargin{i+1};
            case 'low_threshold'
                low_threshold = varargin{i+1};
            case 'high_threshold'
                high_threshold = varargin{i+1};
        end
    end
    
    % Convert to grayscale if needed
    if size(img, 3) > 1
        img = rgb2gray_custom(img);
    end
    
    % Step 1: Gaussian smoothing
    smoothed = apply_gaussian_filter(img, sigma);
    
    % Step 2: Gradient calculation
    [magnitude, direction] = sobel_edge_detection(smoothed);
    
    % Normalize magnitude
    magnitude = magnitude / max(magnitude(:));
    
    % Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, direction);
    
    % Step 4: Double thresholding and hysteresis
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold);
end
