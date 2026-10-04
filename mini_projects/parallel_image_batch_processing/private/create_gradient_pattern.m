% Location: mini_projects/parallel_image_batch_processing/private/create_gradient_pattern.m
% (split out of batch_processor.m so it can be called from other files)

function img = create_gradient_pattern(x, y)
    [h, w] = size(x);
    img = (x + y) * 255 / (h + w);
end
