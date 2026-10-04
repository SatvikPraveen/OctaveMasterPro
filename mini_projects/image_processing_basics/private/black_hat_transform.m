% Location: mini_projects/image_processing_basics/private/black_hat_transform.m
% (split out of morphology_ops.m so it can be called from other files)

function result = black_hat_transform(img, se)
    % Black-hat transform (closing - original)
    
    closed = morphological_closing(img, se);
    result = closed - img;
end
