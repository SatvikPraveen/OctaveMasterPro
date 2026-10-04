% Location: mini_projects/parallel_image_batch_processing/color_correction_operation.m
% (split out of image_operations.m so it can be called from other files)

function processed_img = color_correction_operation(img, varargin)
    % Color correction and enhancement
    
    gamma = 1.0;
    brightness = 0.0;
    contrast = 1.0;
    saturation = 1.0;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'gamma', gamma = varargin{i+1};
            case 'brightness', brightness = varargin{i+1};
            case 'contrast', contrast = varargin{i+1};
            case 'saturation', saturation = varargin{i+1};
        end
    end
    
    processed_img = img;
    
    # Apply gamma correction
    if gamma ~= 1.0
        processed_img = processed_img .^ gamma;
    end
    
    # Apply brightness adjustment
    processed_img = processed_img + brightness;
    
    # Apply contrast adjustment
    if contrast ~= 1.0
        mean_val = mean(processed_img(:));
        processed_img = (processed_img - mean_val) * contrast + mean_val;
    end
    
    # Apply saturation adjustment (for color images)
    if size(processed_img, 3) == 3 && saturation ~= 1.0
        # Convert to HSV-like adjustment
        gray = 0.299*processed_img(:,:,1) + 0.587*processed_img(:,:,2) + 0.114*processed_img(:,:,3);
        
        for c = 1:3
            processed_img(:,:,c) = gray + saturation * (processed_img(:,:,c) - gray);
        end
    end
    
    # Clamp values
    if max(img(:)) <= 1
        processed_img = max(0, min(1, processed_img));
    else
        processed_img = max(0, min(255, processed_img));
    end
end
