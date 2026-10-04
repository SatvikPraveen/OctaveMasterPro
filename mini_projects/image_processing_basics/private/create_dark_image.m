% Location: mini_projects/image_processing_basics/private/create_dark_image.m
% (split out of histogram_analysis.m so it can be called from other files)

function dark_img = create_dark_image()
    % Create dark test image
    [x, y] = meshgrid(1:100, 1:100);
    dark_img = 0.1 + 0.15 * exp(-((x-50).^2 + (y-50).^2)/500);
    dark_img = dark_img + 0.02 * randn(size(x));
    dark_img = max(0, min(1, dark_img));
end
