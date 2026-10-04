% Location: mini_projects/image_processing_basics/private/apply_gaussian_filter.m
% (split out of basic_filters.m so it can be called from other files)

function filtered_img = apply_gaussian_filter(img, sigma, varargin)
    % Apply Gaussian blur filter
    %
    % Inputs:
    %   img - input image
    %   sigma - standard deviation of Gaussian kernel
    %   varargin - optional parameters:
    %     'kernel_size' - size of Gaussian kernel (default: auto)
    %     'separable' - use separable filtering (default: true)
    %
    % Output:
    %   filtered_img - Gaussian filtered image
    
    % Default parameters
    kernel_size = [];
    separable = true;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'kernel_size'
                kernel_size = varargin{i+1};
            case 'separable'
                separable = varargin{i+1};
        end
    end
    
    % Auto-determine kernel size
    if isempty(kernel_size)
        kernel_size = 2 * ceil(3 * sigma) + 1; % 6*sigma + 1 rule
    end
    
    % Ensure odd kernel size
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
    end
    
    if separable
        % Use separable 1D Gaussian filters (more efficient)
        x = -floor(kernel_size/2):floor(kernel_size/2);
        gaussian_1d = exp(-x.^2 / (2*sigma^2));
        gaussian_1d = gaussian_1d / sum(gaussian_1d);
        
        % Apply separable filtering
        if size(img, 3) == 1
            % Grayscale
            filtered_img = conv2(gaussian_1d, gaussian_1d', img, 'same');
        else
            % Color - process each channel
            filtered_img = zeros(size(img));
            for c = 1:size(img, 3)
                filtered_img(:,:,c) = conv2(gaussian_1d, gaussian_1d', img(:,:,c), 'same');
            end
        end
    else
        % Use 2D Gaussian kernel
        [x, y] = meshgrid(-floor(kernel_size/2):floor(kernel_size/2));
        gaussian_2d = exp(-(x.^2 + y.^2) / (2*sigma^2));
        gaussian_2d = gaussian_2d / sum(gaussian_2d(:));
        
        % Apply 2D convolution
        if size(img, 3) == 1
            filtered_img = conv2(img, gaussian_2d, 'same');
        else
            filtered_img = zeros(size(img));
            for c = 1:size(img, 3)
                filtered_img(:,:,c) = conv2(img(:,:,c), gaussian_2d, 'same');
            end
        end
    end
end
