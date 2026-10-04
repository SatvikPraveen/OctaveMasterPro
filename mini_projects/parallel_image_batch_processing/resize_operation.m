% Location: mini_projects/parallel_image_batch_processing/resize_operation.m
% (split out of image_operations.m so it can be called from other files)

function processed_img = resize_operation(img, varargin)
    % Resize image operation for batch processing
    
    target_size = [256, 256];
    method = 'bilinear';
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'size', target_size = varargin{i+1};
            case 'method', method = varargin{i+1};
        end
    end
    
    [h, w, c] = size(img);
    
    if h == target_size(1) && w == target_size(2)
        processed_img = img;
        return;
    end
    
    # Simple bilinear interpolation resize
    new_h = target_size(1);
    new_w = target_size(2);
    
    [X_old, Y_old] = meshgrid(1:w, 1:h);
    [X_new, Y_new] = meshgrid(linspace(1, w, new_w), linspace(1, h, new_h));
    
    if c == 1
        processed_img = interp2(X_old, Y_old, img, X_new, Y_new, 'linear', 0);
    else
        processed_img = zeros(new_h, new_w, c);
        for ch = 1:c
            processed_img(:,:,ch) = interp2(X_old, Y_old, img(:,:,ch), X_new, Y_new, 'linear', 0);
        end
    end
end
