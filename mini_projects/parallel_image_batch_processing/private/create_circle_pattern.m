% Location: mini_projects/parallel_image_batch_processing/private/create_circle_pattern.m
% (split out of batch_processor.m so it can be called from other files)

function img = create_circle_pattern(x, y)
    img = zeros(size(x));
    [h, w] = size(x);
    
    num_circles = 3 + round(3 * rand());
    for c = 1:num_circles
        center_x = round(w * rand());
        center_y = round(h * rand());
        radius = 10 + round(20 * rand());
        intensity = 100 + 155 * rand();
        
        circle_mask = (x - center_x).^2 + (y - center_y).^2 <= radius^2;
        img = img + circle_mask * intensity;
    end
    
    img = min(img, 255);
end
