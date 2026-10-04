% Location: mini_projects/parallel_image_batch_processing/image_operations.m
% Core Image Processing Functions for Batch Processing
%
% The batch operations (resize_operation, filter_operation,
% enhance_operation, enhance_contrast_operation, normalize_operation,
% color_correction_operation, noise_reduction_operation,
% create_composite_operation, create_thumbnail, add_watermark) are
% separate files in this directory; their internal helpers live in
% private/.  Calling image_operations() runs the image operations demo.

function image_operations()
    demo_image_operations();
end
