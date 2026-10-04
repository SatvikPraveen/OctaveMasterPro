% Location: mini_projects/parallel_image_batch_processing/filter_operation.m
% (split out of image_operations.m so it can be called from other files)

function processed_img = filter_operation(img, varargin)
    % Apply filtering operations
    
    filter_type = 'gaussian';
    sigma = 1.5;
    kernel_size = 5;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'type', filter_type = varargin{i+1};
            case 'sigma', sigma = varargin{i+1};
            case 'kernel_size', kernel_size = varargin{i+1};
        end
    end
    
    switch lower(filter_type)
        case 'gaussian'
            processed_img = apply_gaussian_blur(img, sigma);
        case 'median'
            processed_img = apply_median_filter_fast(img, kernel_size);
        case 'sharpen'
            processed_img = apply_sharpening_filter(img);
        case 'edge'
            processed_img = apply_edge_detection(img);
        otherwise
            processed_img = img;
    end
end
