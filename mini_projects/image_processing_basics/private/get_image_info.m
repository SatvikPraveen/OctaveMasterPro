% Location: mini_projects/image_processing_basics/private/get_image_info.m
% (split out of image_loader.m so it can be called from other files)

function info = get_image_info(img)
    % Get comprehensive information about an image
    
    [h, w, c] = size(img);
    
    info.height = h;
    info.width = w;
    info.channels = c;
    info.total_pixels = h * w;
    info.data_type = class(img);
    
    if max(img(:)) <= 1
        info.range = [0, 1];
        info.normalized = true;
    else
        info.range = [0, 255];
        info.normalized = false;
    end
    
    info.min_value = min(img(:));
    info.max_value = max(img(:));
    info.mean_value = mean(img(:));
    info.std_value = std(img(:));
    
    if c == 1
        info.type = 'grayscale';
    else
        info.type = 'color';
    end
    
    % Display information
    fprintf('\nImage Information:\n');
    fprintf('------------------\n');
    fprintf('Dimensions: %d x %d x %d\n', h, w, c);
    fprintf('Type: %s\n', info.type);
    fprintf('Data type: %s\n', info.data_type);
    fprintf('Value range: [%.3f, %.3f]\n', info.min_value, info.max_value);
    fprintf('Mean: %.3f, Std: %.3f\n', info.mean_value, info.std_value);
    fprintf('Total pixels: %d\n', info.total_pixels);
end
