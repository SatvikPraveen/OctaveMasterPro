% Location: mini_projects/image_processing_basics/private/enhance_image.m
% (split out of image_loader.m so it can be called from other files)

function enhanced_img = enhance_image(img)
    % Apply basic image enhancement
    
    enhanced_img = img;
    
    % Contrast enhancement using histogram stretching
    if size(img, 3) == 1 % Grayscale
        min_val = min(img(:));
        max_val = max(img(:));
        if max_val > min_val
            enhanced_img = (img - min_val) / (max_val - min_val);
        end
    else % Color
        for c = 1:size(img, 3)
            channel = img(:,:,c);
            min_val = min(channel(:));
            max_val = max(channel(:));
            if max_val > min_val
                enhanced_img(:,:,c) = (channel - min_val) / (max_val - min_val);
            end
        end
    end
    
    % Slight sharpening
    if size(enhanced_img, 3) == 1
        sharp_kernel = [0 -0.5 0; -0.5 3 -0.5; 0 -0.5 0];
        enhanced_img = conv2(enhanced_img, sharp_kernel, 'same');
        enhanced_img = max(0, min(1, enhanced_img)); % Clamp to [0,1]
    end
end
