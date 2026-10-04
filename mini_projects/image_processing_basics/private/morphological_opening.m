% Location: mini_projects/image_processing_basics/private/morphological_opening.m
% (split out of morphology_ops.m so it can be called from other files)

function result = morphological_opening(img, se)
    % Morphological opening (erosion followed by dilation)
    
    eroded = morphological_erosion(img, se);
    result = morphological_dilation(eroded, se);
end
