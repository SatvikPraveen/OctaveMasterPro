% Location: mini_projects/image_processing_basics/private/resize_image.m
% (split out of image_loader.m so it can be called from other files)

function resized_img = resize_image(img, new_dims)
    % Simple bilinear interpolation resize
    
    [old_h, old_w, channels] = size(img);
    new_h = new_dims(1);
    new_w = new_dims(2);
    
    % Create coordinate grids
    [X_old, Y_old] = meshgrid(1:old_w, 1:old_h);
    [X_new, Y_new] = meshgrid(linspace(1, old_w, new_w), linspace(1, old_h, new_h));
    
    % Interpolate each channel
    if channels == 1
        resized_img = interp2(X_old, Y_old, img, X_new, Y_new, 'linear', 0);
    else
        resized_img = zeros(new_h, new_w, channels);
        for c = 1:channels
            resized_img(:,:,c) = interp2(X_old, Y_old, img(:,:,c), X_new, Y_new, 'linear', 0);
        end
    end
end
