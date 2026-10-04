% Location: mini_projects/parallel_image_batch_processing/private/create_noise_pattern.m
% (split out of batch_processor.m so it can be called from other files)

function img = create_noise_pattern(img_size)
    img = 127 + 50 * randn(img_size, img_size);
end
