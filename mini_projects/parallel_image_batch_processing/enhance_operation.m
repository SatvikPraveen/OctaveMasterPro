% Location: mini_projects/parallel_image_batch_processing/enhance_operation.m
% (split out of image_operations.m so it can be called from other files)

function enhanced_img = enhance_operation(img, varargin)
    % Comprehensive image enhancement
    
    enhance_contrast = true;
    enhance_sharpness = true;
    denoise = true;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'contrast', enhance_contrast = varargin{i+1};
            case 'sharpness', enhance_sharpness = varargin{i+1};
            case 'denoise', denoise = varargin{i+1};
        end
    end
    
    enhanced_img = img;
    
    # Denoising
    if denoise
        enhanced_img = apply_gaussian_blur(enhanced_img, 0.8);
    end
    
    # Contrast enhancement
    if enhance_contrast
        enhanced_img = enhance_contrast_operation(enhanced_img);
    end
    
    # Sharpening
    if enhance_sharpness
        enhanced_img = apply_sharpening_filter(enhanced_img);
    end
end
