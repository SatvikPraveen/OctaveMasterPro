% Location: mini_projects/parallel_image_batch_processing/noise_reduction_operation.m
% (split out of image_operations.m so it can be called from other files)

function processed_img = noise_reduction_operation(img, varargin)
    % Noise reduction for batch processing
    
    method = 'bilateral';
    strength = 1.0;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'method', method = varargin{i+1};
            case 'strength', strength = varargin{i+1};
        end
    end
    
    switch lower(method)
        case 'gaussian'
            processed_img = apply_gaussian_blur(img, 1.0 * strength);
            
        case 'median'
            kernel_size = round(3 * strength);
            if mod(kernel_size, 2) == 0, kernel_size = kernel_size + 1; end
            processed_img = apply_median_filter_fast(img, kernel_size);
            
        case 'bilateral'
            processed_img = apply_simple_bilateral(img, strength);
            
        otherwise
            processed_img = img;
    end
end
