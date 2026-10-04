% Location: mini_projects/image_processing_basics/private/create_low_contrast_image.m
% (split out of histogram_analysis.m so it can be called from other files)

function low_img = create_low_contrast_image()
    % Create low contrast test image
    [x, y] = meshgrid(1:100, 1:100);
    low_img = 0.4 + 0.2 * sin(x/10) .* cos(y/8);
    low_img = low_img + 0.05 * randn(size(x));
    low_img = max(0, min(1, low_img));
end
