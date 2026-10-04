% Location: mini_projects/parallel_image_batch_processing/private/apply_edge_detection.m
% (split out of image_operations.m so it can be called from other files)

function edges = apply_edge_detection(img)
    % Simple edge detection for batch processing
    
    if size(img, 3) > 1
        gray_img = 0.299*img(:,:,1) + 0.587*img(:,:,2) + 0.114*img(:,:,3);
    else
        gray_img = img;
    end
    
    # Sobel operators
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    
    grad_x = conv2(gray_img, sobel_x, 'same');
    grad_y = conv2(gray_img, sobel_y, 'same');
    
    edges = sqrt(grad_x.^2 + grad_y.^2);
    
    # Normalize
    edges = edges / max(edges(:));
end
