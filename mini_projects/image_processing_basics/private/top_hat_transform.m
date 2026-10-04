% Location: mini_projects/image_processing_basics/private/top_hat_transform.m
% (split out of morphology_ops.m so it can be called from other files)

function result = top_hat_transform(img, se)
    % Top-hat transform (original - opening)
    
    opened = morphological_opening(img, se);
    result = img - opened;
end
