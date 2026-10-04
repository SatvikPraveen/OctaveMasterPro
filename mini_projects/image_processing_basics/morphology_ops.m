% Location: mini_projects/image_processing_basics/morphology_ops.m
% Mathematical Morphology Operations
%
% The operations (create_structuring_element, morphological_erosion,
% morphological_dilation, morphological_opening, morphological_closing,
% morphological_gradient, top_hat_transform, black_hat_transform,
% extract_boundary, morphological_skeletonization, ...) live in private/
% so that image_demo and the other files of this project can call them.
% Calling morphology_ops() runs the basic and advanced morphology demos.

function morphology_ops()
    demo_morphological_operations();
    demo_advanced_morphology();
end
