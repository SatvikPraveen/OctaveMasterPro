% Location: mini_projects/image_processing_basics/private/create_complex_binary_image.m
% (split out of morphology_ops.m so it can be called from other files)

function complex_img = create_complex_binary_image()
    % Create complex binary image for advanced demonstrations
    
    img = zeros(150, 150);
    
    % Various shapes and structures
    [x, y] = meshgrid(1:150, 1:150);
    
    % Large circle
    img = img + double((x-40).^2 + (y-40).^2 < 25^2);
    
    % Rectangle with hole
    img(80:120, 80:120) = 1;
    img(90:110, 90:110) = 0;
    
    % Connected components
    img(20:30, 100:140) = 1;
    img(25:35, 120:130) = 1;
    
    % Thin structures
    img(60:90, 65:67) = 1;
    img(100:102, 30:60) = 1;
    
    complex_img = img > 0;
end
