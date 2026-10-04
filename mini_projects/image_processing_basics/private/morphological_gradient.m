% Location: mini_projects/image_processing_basics/private/morphological_gradient.m
% (split out of morphology_ops.m so it can be called from other files)

function result = morphological_gradient(img, se)
    % Morphological gradient (dilation - erosion)
    
    dilated = morphological_dilation(img, se);
    eroded = morphological_erosion(img, se);
    result = dilated - eroded;
end
