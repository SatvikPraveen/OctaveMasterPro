% Location: mini_projects/parallel_image_batch_processing/private/create_stripe_pattern.m
% (split out of batch_processor.m so it can be called from other files)

function img = create_stripe_pattern(x, y, img_size)
    freq = 0.1 + 0.3 * rand();
    angle = 180 * rand();
    
    rotated_x = x * cos(angle*pi/180) - y * sin(angle*pi/180);
    img = 127 + 127 * sin(2*pi*freq*rotated_x/img_size);
end
