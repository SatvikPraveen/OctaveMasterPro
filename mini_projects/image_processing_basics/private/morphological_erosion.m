% Location: mini_projects/image_processing_basics/private/morphological_erosion.m
% (split out of morphology_ops.m so it can be called from other files)

function result = morphological_erosion(img, se)
    % Morphological erosion operation
    
    if max(img(:)) > 1, img = img > 0.5; end
    
    [h, w] = size(img);
    [se_h, se_w] = size(se);
    pad_h = floor(se_h / 2);
    pad_w = floor(se_w / 2);
    padded_img = padarray(img, [pad_h, pad_w], 0);
    
    result = zeros(h, w);
    
    for i = 1:h
        for j = 1:w
            window = padded_img(i:i+se_h-1, j:j+se_w-1);
            masked_window = window(se > 0);
            result(i, j) = min(masked_window);
        end
    end
end
