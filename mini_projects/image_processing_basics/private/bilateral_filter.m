% Location: mini_projects/image_processing_basics/private/bilateral_filter.m
% (split out of basic_filters.m so it can be called from other files)

function filtered_img = bilateral_filter(img, sigma_spatial, sigma_intensity, varargin)
    % Bilateral filter for edge-preserving smoothing
    %
    % Inputs:
    %   img - input image
    %   sigma_spatial - spatial standard deviation
    %   sigma_intensity - intensity standard deviation
    %   varargin - optional parameters:
    %     'kernel_size' - filter window size
    %
    % Output:
    %   filtered_img - bilateral filtered image
    
    % Default parameters
    kernel_size = 2 * ceil(3 * sigma_spatial) + 1;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'kernel_size'
                kernel_size = varargin{i+1};
        end
    end
    
    % Ensure odd kernel size
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
    end
    
    [h, w, c] = size(img);
    filtered_img = zeros(size(img));
    pad_size = floor(kernel_size / 2);
    
    % Create spatial weight matrix
    [x, y] = meshgrid(-pad_size:pad_size);
    spatial_weights = exp(-(x.^2 + y.^2) / (2 * sigma_spatial^2));
    
    % Process each channel
    for ch = 1:c
        channel = img(:,:,ch);
        padded_channel = padarray(channel, [pad_size, pad_size], 'replicate');
        
        for i = 1:h
            for j = 1:w
                % Extract neighborhood
                neighborhood = padded_channel(i:i+kernel_size-1, j:j+kernel_size-1);
                center_value = channel(i, j);
                
                % Compute intensity weights
                intensity_diff = neighborhood - center_value;
                intensity_weights = exp(-intensity_diff.^2 / (2 * sigma_intensity^2));
                
                % Combine spatial and intensity weights
                combined_weights = spatial_weights .* intensity_weights;
                combined_weights = combined_weights / sum(combined_weights(:));
                
                % Apply bilateral filter
                filtered_img(i, j, ch) = sum(sum(neighborhood .* combined_weights));
            end
        end
    end
end
