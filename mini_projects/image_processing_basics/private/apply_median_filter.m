% Location: mini_projects/image_processing_basics/private/apply_median_filter.m
% (split out of basic_filters.m so it can be called from other files)

function filtered_img = apply_median_filter(img, kernel_size)
    % Apply median filter for noise reduction
    %
    % Inputs:
    %   img - input image
    %   kernel_size - size of median filter window (odd number)
    %
    % Output:
    %   filtered_img - median filtered image
    
    if mod(kernel_size, 2) == 0
        kernel_size = kernel_size + 1;
        fprintf('Kernel size adjusted to %d (must be odd)\n', kernel_size);
    end
    
    [h, w, c] = size(img);
    filtered_img = zeros(size(img));
    pad_size = floor(kernel_size / 2);
    
    % Pad image
    if c == 1
        padded_img = padarray(img, [pad_size, pad_size], 'replicate');
    else
        padded_img = zeros(h + 2*pad_size, w + 2*pad_size, c);
        for ch = 1:c
            padded_img(:,:,ch) = padarray(img(:,:,ch), [pad_size, pad_size], 'replicate');
        end
    end
    
    % Apply median filter
    for i = 1:h
        for j = 1:w
            for ch = 1:c
                window = padded_img(i:i+kernel_size-1, j:j+kernel_size-1, ch);
                filtered_img(i, j, ch) = median(window(:));
            end
        end
    end
end
