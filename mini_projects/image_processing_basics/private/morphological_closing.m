% Location: mini_projects/image_processing_basics/private/morphological_closing.m
% (split out of morphology_ops.m so it can be called from other files)

function result = morphological_closing(img, se)
    % Morphological closing (dilation followed by erosion)
    
    dilated = morphological_dilation(img, se);
    result = morphological_erosion(dilated, se);
end
