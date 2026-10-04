% Location: mini_projects/image_processing_basics/basic_filters.m
% Basic Image Filtering Operations
%
% The filters (apply_gaussian_filter, apply_median_filter,
% sobel_edge_detection, canny_edge_detection, unsharp_masking,
% bilateral_filter, calculate_psnr, ...) live in private/ so that
% image_demo and the other files of this project can call them.
% Calling basic_filters() runs the basic and advanced filter demos.

function basic_filters()
    demo_basic_filters();
    demo_advanced_filters();
end
