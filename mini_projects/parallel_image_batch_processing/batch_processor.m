% Location: mini_projects/parallel_image_batch_processing/batch_processor.m
% Main Parallel Image Batch Processing Controller
%
% The controller is process_image_batch.m; its helpers
% (get_image_file_list, generate_test_images, parallel_image_processing,
% sequential_image_processing, benchmark_processing_modes, ...) live in
% private/.  batch_processor(...) is a thin alias for
% process_image_batch(...).

function results = batch_processor(varargin)
    results = process_image_batch(varargin{:});
end
