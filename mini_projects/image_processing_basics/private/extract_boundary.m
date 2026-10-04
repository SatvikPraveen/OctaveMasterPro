% Location: mini_projects/image_processing_basics/private/extract_boundary.m
% (split out of morphology_ops.m so it can be called from other files)

function boundary = extract_boundary(img, se)
    % Extract object boundaries using morphological operations
    
    if max(img(:)) > 1, img = img > 0.5; end
    
    % Boundary = Original - Erosion
    eroded = morphological_erosion(img, se);
    boundary = img - eroded;
end
