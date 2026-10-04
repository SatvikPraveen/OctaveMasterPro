% Location: mini_projects/image_processing_basics/private/unsharp_masking.m
% (split out of basic_filters.m so it can be called from other files)

function filtered_img = unsharp_masking(img, varargin)
    % Unsharp masking for image sharpening
    %
    % Inputs:
    %   img - input image
    %   varargin - optional parameters:
    %     'sigma' - Gaussian blur sigma (default: 1)
    %     'strength' - sharpening strength (default: 1.5)
    %
    % Output:
    %   filtered_img - sharpened image
    
    % Default parameters
    sigma = 1;
    strength = 1.5;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'sigma'
                sigma = varargin{i+1};
            case 'strength'
                strength = varargin{i+1};
        end
    end
    
    % Create unsharp mask
    blurred = apply_gaussian_filter(img, sigma);
    mask = img - blurred;
    
    % Apply sharpening
    filtered_img = img + strength * mask;
    
    % Clamp values
    if max(img(:)) <= 1
        filtered_img = max(0, min(1, filtered_img));
    else
        filtered_img = max(0, min(255, filtered_img));
    end
end
