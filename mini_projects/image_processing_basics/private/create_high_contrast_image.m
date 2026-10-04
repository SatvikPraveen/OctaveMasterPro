% Location: mini_projects/image_processing_basics/private/create_high_contrast_image.m
% (split out of histogram_analysis.m so it can be called from other files)

function high_img = create_high_contrast_image()
    % Create high contrast test image
    high_img = zeros(100, 100);
    high_img(20:40, 20:80) = 1;
    high_img(60:80, 20:40) = 1;
    high_img(60:80, 60:80) = 1;
end
