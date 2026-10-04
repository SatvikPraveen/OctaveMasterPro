% Location: mini_projects/image_processing_basics/private/morphological_skeletonization.m
% (split out of morphology_ops.m so it can be called from other files)

function skeleton = morphological_skeletonization(img)
    % Simple skeletonization using iterative erosion
    
    if max(img(:)) > 1, img = img > 0.5; end
    
    se = create_structuring_element('cross', 1);
    skeleton = zeros(size(img));
    current = img;
    
    while sum(current(:)) > 0
        eroded = morphological_erosion(current, se);
        opened = morphological_opening(eroded, se);
        skeleton = skeleton | (eroded - opened);
        current = eroded;
    end
end
