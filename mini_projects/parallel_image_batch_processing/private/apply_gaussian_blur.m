% Location: mini_projects/parallel_image_batch_processing/private/apply_gaussian_blur.m
% (split out of image_operations.m so it can be called from other files)

function blurred_img = apply_gaussian_blur(img, sigma)
    % Fast Gaussian blur implementation
    
    kernel_size = 2 * ceil(3 * sigma) + 1;
    
    # Create 1D Gaussian kernel
    x = -floor(kernel_size/2):floor(kernel_size/2);
    gaussian_1d = exp(-x.^2 / (2*sigma^2));
    gaussian_1d = gaussian_1d / sum(gaussian_1d);
    
    # Apply separable filtering
    if size(img, 3) == 1
        blurred_img = conv2(gaussian_1d, gaussian_1d', img, 'same');
    else
        blurred_img = zeros(size(img));
        for c = 1:size(img, 3)
            blurred_img(:,:,c) = conv2(gaussian_1d, gaussian_1d', img(:,:,c), 'same');
        end
    end
end
