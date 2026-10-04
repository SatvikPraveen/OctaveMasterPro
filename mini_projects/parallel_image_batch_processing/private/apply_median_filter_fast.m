% Location: mini_projects/parallel_image_batch_processing/private/apply_median_filter_fast.m
% (split out of image_operations.m so it can be called from other files)

function filtered_img = apply_median_filter_fast(img, kernel_size)
    % Fast median filter implementation
    
    [h, w, c] = size(img);
    filtered_img = zeros(size(img));
    pad_size = floor(kernel_size / 2);
    
    for ch = 1:c
        channel = img(:,:,ch);
        padded = padarray(channel, [pad_size, pad_size], 'replicate');
        
        for i = 1:h
            for j = 1:w
                window = padded(i:i+kernel_size-1, j:j+kernel_size-1);
                filtered_img(i, j, ch) = median(window(:));
            end
        end
    end
end
