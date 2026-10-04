% Location: mini_projects/image_processing_basics/private/calculate_psnr.m
% (split out of basic_filters.m so it can be called from other files)

function psnr_val = calculate_psnr(original, processed)
    % Calculate Peak Signal-to-Noise Ratio
    
    mse = mean((original(:) - processed(:)).^2);
    if mse == 0
        psnr_val = inf;
    else
        max_val = max(original(:));
        psnr_val = 20 * log10(max_val / sqrt(mse));
    end
end
