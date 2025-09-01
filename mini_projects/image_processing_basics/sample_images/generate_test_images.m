% Location: mini_projects/image_processing_basics/sample_images/generate_test_images.m
% Generate Test Images for Image Processing Demonstrations

function generate_test_images()
    fprintf('Generating test images...\n');
    
    images_dir = fileparts(mfilename('fullpath'));
    if ~exist(images_dir, 'dir')
        mkdir(images_dir);
    end
    
    generate_lena_synthetic(images_dir);
    generate_geometric_shapes(images_dir);
    generate_texture_patterns(images_dir);
    generate_noisy_images(images_dir);
    generate_contrast_test_images(images_dir);
    generate_frequency_test_images(images_dir);
    generate_color_test_images(images_dir);
    generate_morphology_test_images(images_dir);
    
    fprintf('Test image generation complete!\n');
    fprintf('Generated files in: %s\n', images_dir);
end

function generate_lena_synthetic(images_dir)
    fprintf('  Generating synthetic Lena image...\n');
    
    img_size = 512;
    img = zeros(img_size, img_size, 3);
    
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # Face region
    face_center_x = img_size * 0.5;
    face_center_y = img_size * 0.45;
    face_width = img_size * 0.25;
    face_height = img_size * 0.35;
    
    face_mask = ((x - face_center_x) / face_width).^2 + ((y - face_center_y) / face_height).^2 <= 1;
    
    img(:,:,1) = img(:,:,1) + face_mask * 0.9;
    img(:,:,2) = img(:,:,2) + face_mask * 0.7;
    img(:,:,3) = img(:,:,3) + face_mask * 0.6;
    
    # Hair region
    hair_mask = y < img_size * 0.3 | (y < img_size * 0.5 & abs(x - img_size/2) > img_size * 0.2);
    img(:,:,1) = img(:,:,1) + hair_mask * 0.2;
    img(:,:,2) = img(:,:,2) + hair_mask * 0.1;
    img(:,:,3) = img(:,:,3) + hair_mask * 0.05;
    
    # Eyes
    left_eye_x = img_size * 0.42;
    left_eye_y = img_size * 0.38;
    right_eye_x = img_size * 0.58;
    right_eye_y = img_size * 0.38;
    eye_radius = img_size * 0.02;
    
    left_eye_mask = (x - left_eye_x).^2 + (y - left_eye_y).^2 <= eye_radius^2;
    right_eye_mask = (x - right_eye_x).^2 + (y - right_eye_y).^2 <= eye_radius^2;
    
    img(left_eye_mask | right_eye_mask) = 0;
    
    # Mouth
    mouth_center_x = img_size * 0.5;
    mouth_center_y = img_size * 0.6;
    mouth_width = img_size * 0.04;
    mouth_height = img_size * 0.01;
    
    mouth_mask = ((x - mouth_center_x) / mouth_width).^2 + ((y - mouth_center_y) / mouth_height).^2 <= 1;
    img(mouth_mask) = 0.3;
    
    # Add texture and lighting
    for c = 1:3
        lighting = 0.1 * sin(pi * x / img_size) .* cos(pi * y / img_size);
        img(:,:,c) = img(:,:,c) + lighting + 0.02 * randn(img_size);
    end
    
    img = max(0, min(1, img));
    
    imwrite(img, fullfile(images_dir, 'lena_synthetic.png'));
    
    gray_img = 0.299*img(:,:,1) + 0.587*img(:,:,2) + 0.114*img(:,:,3);
    imwrite(gray_img, fullfile(images_dir, 'lena_synthetic_gray.png'));
end

function generate_geometric_shapes(images_dir)
    fprintf('  Generating geometric shapes...\n');
    
    img_size = 400;
    img = zeros(img_size, img_size, 3);
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # Circle (red)
    circle_center = [100, 100];
    circle_radius = 60;
    circle_mask = (x - circle_center(1)).^2 + (y - circle_center(2)).^2 <= circle_radius^2;
    img(circle_mask) = 1;
    img(:,:,2) = img(:,:,2) .* ~circle_mask;
    img(:,:,3) = img(:,:,3) .* ~circle_mask;
    
    # Rectangle (green)
    rect_x = [250, 350];
    rect_y = [50, 150];
    rect_mask = x >= rect_x(1) & x <= rect_x(2) & y >= rect_y(1) & y <= rect_y(2);
    img(:,:,2) = img(:,:,2) + rect_mask;
    
    # Triangle (blue)
    tri_vertices = [80, 300; 180, 300; 130, 200];
    tri_mask = inpolygon(x, y, tri_vertices(:,1), tri_vertices(:,2));
    img(:,:,3) = img(:,:,3) + tri_mask;
    
    # Ellipse (yellow)
    ellipse_center = [300, 280];
    ellipse_a = 50;
    ellipse_b = 30;
    ellipse_mask = ((x - ellipse_center(1)) / ellipse_a).^2 + ((y - ellipse_center(2)) / ellipse_b).^2 <= 1;
    img(:,:,1) = img(:,:,1) + ellipse_mask * 0.8;
    img(:,:,2) = img(:,:,2) + ellipse_mask * 0.8;
    
    # Small circles
    small_circles_x = [150, 200, 250, 300];
    small_circles_y = [350, 320, 350, 320];
    
    for i = 1:length(small_circles_x)
        small_circle_mask = (x - small_circles_x(i)).^2 + (y - small_circles_y(i)).^2 <= 15^2;
        color_intensity = 0.3 + 0.4 * i / length(small_circles_x);
        img(:,:,mod(i-1, 3)+1) = img(:,:,mod(i-1, 3)+1) + small_circle_mask * color_intensity;
    end
    
    img = max(0, min(1, img));
    imwrite(img, fullfile(images_dir, 'geometric_shapes.png'));
end

function generate_texture_patterns(images_dir)
    fprintf('  Generating texture patterns...\n');
    
    img_size = 512;
    [x, y] = meshgrid(1:img_size, 1:img_size);
    texture_img = zeros(img_size, img_size);
    
    # Quadrant 1: Sinusoidal pattern
    quad1_mask = x <= img_size/2 & y <= img_size/2;
    texture_img(quad1_mask) = 0.5 + 0.3 * sin(2*pi*x(quad1_mask)/30) .* cos(2*pi*y(quad1_mask)/30);
    
    # Quadrant 2: Checkerboard
    quad2_mask = x > img_size/2 & y <= img_size/2;
    checker_size = 32;
    checker_x = floor(x / checker_size);
    checker_y = floor(y / checker_size);
    checkerboard = mod(checker_x + checker_y, 2);
    texture_img(quad2_mask) = checkerboard(quad2_mask);
    
    # Quadrant 3: Random noise texture
    quad3_mask = x <= img_size/2 & y > img_size/2;
    noise_texture = 0.5 + 0.2 * randn(img_size);
    texture_img(quad3_mask) = noise_texture(quad3_mask);
    
    # Quadrant 4: Concentric circles
    quad4_mask = x > img_size/2 & y > img_size/2;
    center_x = 3*img_size/4;
    center_y = 3*img_size/4;
    radius = sqrt((x - center_x).^2 + (y - center_y).^2);
    concentric = 0.5 + 0.3 * sin(radius / 10);
    texture_img(quad4_mask) = concentric(quad4_mask);
    
    texture_img = max(0, min(1, texture_img));
    
    imwrite(texture_img, fullfile(images_dir, 'texture_patterns.png'));
    
    # Color version
    texture_color = zeros(img_size, img_size, 3);
    texture_color(:,:,1) = texture_img;
    texture_color(:,:,2) = circshift(texture_img, [0, img_size/4]);
    texture_color(:,:,3) = circshift(texture_img, [img_size/4, 0]);
    
    imwrite(texture_color, fullfile(images_dir, 'texture_patterns_color.png'));
end

function generate_noisy_images(images_dir)
    fprintf('  Generating noisy images...\n');
    
    img_size = 256;
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # Clean base image
    clean_img = zeros(img_size, img_size);
    clean_img = clean_img + double((x - img_size/3).^2 + (y - img_size/3).^2 <= 40^2) * 0.8;
    clean_img = clean_img + double((x - 2*img_size/3).^2 + (y - 2*img_size/3).^2 <= 30^2) * 0.6;
    clean_img = clean_img + 0.3 * sin(2*pi*x/50) .* cos(2*pi*y/50);
    clean_img = max(0, min(1, clean_img));
    
    # Gaussian noise
    gaussian_noise_img = clean_img + 0.1 * randn(img_size);
    gaussian_noise_img = max(0, min(1, gaussian_noise_img));
    
    # Salt and pepper noise
    salt_pepper_img = clean_img;
    noise_density = 0.05;
    salt_locations = rand(img_size) < noise_density/2;
    pepper_locations = rand(img_size) < noise_density/2;
    salt_pepper_img(salt_locations) = 1;
    salt_pepper_img(pepper_locations) = 0;
    
    # Speckle noise
    speckle_img = clean_img .* (1 + 0.2 * randn(img_size));
    speckle_img = max(0, min(1, speckle_img));
    
    # Save images
    imwrite(clean_img, fullfile(images_dir, 'clean_test_image.png'));
    imwrite(gaussian_noise_img, fullfile(images_dir, 'gaussian_noise.png'));
    imwrite(salt_pepper_img, fullfile(images_dir, 'salt_pepper_noise.png'));
    imwrite(speckle_img, fullfile(images_dir, 'speckle_noise.png'));
end

function generate_contrast_test_images(images_dir)
    fprintf('  Generating contrast test images...\n');
    
    img_size = 300;
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # Base image with circles
    base_img = zeros(img_size, img_size);
    intensities = [0.2, 0.4, 0.6, 0.8, 1.0];
    centers_x = [60, 150, 240, 120, 180];
    centers_y = [60, 60, 60, 180, 240];
    radius = 40;
    
    for i = 1:length(intensities)
        circle_mask = (x - centers_x(i)).^2 + (y - centers_y(i)).^2 <= radius^2;
        base_img(circle_mask) = intensities(i);
    end
    
    gradient = 0.1 * (x + y) / (2 * img_size);
    base_img = base_img + gradient;
    base_img = max(0, min(1, base_img));
    
    # Variations
    low_contrast = 0.3 + 0.4 * base_img;
    high_contrast = base_img;
    high_contrast(base_img < 0.5) = high_contrast(base_img < 0.5) * 0.5;
    high_contrast(base_img >= 0.5) = 0.5 + (high_contrast(base_img >= 0.5) - 0.5) * 2;
    high_contrast = max(0, min(1, high_contrast));
    very_low_contrast = 0.45 + 0.1 * base_img;
    
    imwrite(base_img, fullfile(images_dir, 'normal_contrast.png'));
    imwrite(low_contrast, fullfile(images_dir, 'low_contrast.png'));
    imwrite(high_contrast, fullfile(images_dir, 'high_contrast.png'));
    imwrite(very_low_contrast, fullfile(images_dir, 'very_low_contrast.png'));
end

function generate_frequency_test_images(images_dir)
    fprintf('  Generating frequency test images...\n');
    
    img_size = 256;
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # High frequency patterns
    high_freq = 0.5 + 0.3 * sin(2*pi*x/8) .* cos(2*pi*y/8);
    high_freq = max(0, min(1, high_freq));
    
    # Low frequency patterns
    low_freq = 0.5 + 0.3 * sin(2*pi*x/64) .* cos(2*pi*y/64);
    low_freq = max(0, min(1, low_freq));
    
    # Mixed frequency patterns
    mixed_freq = 0.3 * sin(2*pi*x/8) .* cos(2*pi*y/8) + ...
                0.4 * sin(2*pi*x/32) .* cos(2*pi*y/32) + ...
                0.3 * sin(2*pi*x/128) .* cos(2*pi*y/128);
    mixed_freq = 0.5 + mixed_freq;
    mixed_freq = max(0, min(1, mixed_freq));
    
    # Radial frequency pattern
    center = img_size / 2;
    radius = sqrt((x - center).^2 + (y - center).^2);
    radial_freq = 0.5 + 0.3 * sin(2*pi*radius/16);
    radial_freq = max(0, min(1, radial_freq));
    
    imwrite(high_freq, fullfile(images_dir, 'high_frequency.png'));
    imwrite(low_freq, fullfile(images_dir, 'low_frequency.png'));
    imwrite(mixed_freq, fullfile(images_dir, 'mixed_frequency.png'));
    imwrite(radial_freq, fullfile(images_dir, 'radial_frequency.png'));
end

function generate_color_test_images(images_dir)
    fprintf('  Generating color test images...\n');
    
    img_size = 300;
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # RGB color gradient
    rgb_gradient = zeros(img_size, img_size, 3);
    rgb_gradient(:,:,1) = x / img_size;
    rgb_gradient(:,:,2) = y / img_size;
    rgb_gradient(:,:,3) = (x + y) / (2 * img_size);
    
    # HSV color wheel
    center_x = img_size / 2;
    center_y = img_size / 2;
    
    hsv_img = zeros(img_size, img_size, 3);
    
    for i = 1:img_size
        for j = 1:img_size
            dx = j - center_x;
            dy = i - center_y;
            radius = sqrt(dx^2 + dy^2);
            angle = atan2(dy, dx);
            
            if radius <= img_size/2 * 0.8
                # Hue from angle
                hue = (angle + pi) / (2*pi);
                # Saturation from radius
                saturation = radius / (img_size/2 * 0.8);
                # Value constant
                value = 0.9;
                
                # Convert HSV to RGB (simplified)
                c = value * saturation;
                x_val = c * (1 - abs(mod(hue * 6, 2) - 1));
                m = value - c;
                
                if hue < 1/6
                    hsv_img(i, j, :) = [c, x_val, 0] + m;
                elseif hue < 2/6
                    hsv_img(i, j, :) = [x_val, c, 0] + m;
                elseif hue < 3/6
                    hsv_img(i, j, :) = [0, c, x_val] + m;
                elseif hue < 4/6
                    hsv_img(i, j, :) = [0, x_val, c] + m;
                elseif hue < 5/6
                    hsv_img(i, j, :) = [x_val, 0, c] + m;
                else
                    hsv_img(i, j, :) = [c, 0, x_val] + m;
                end
            end
        end
    end
    
    # Primary colors test
    primary_colors = zeros(img_size, img_size, 3);
    
    # Red square
    primary_colors(50:100, 50:100, 1) = 1;
    # Green square
    primary_colors(50:100, 150:200, 2) = 1;
    # Blue square
    primary_colors(150:200, 50:100, 3) = 1;
    # Cyan square (green + blue)
    primary_colors(150:200, 150:200, 2:3) = 1;
    # Magenta square (red + blue)
    primary_colors(50:100, 250:300, [1,3]) = 1;
    # Yellow square (red + green)
    primary_colors(150:200, 250:300, 1:2) = 1;
    # White square (all colors)
    primary_colors(250:300, 150:200, :) = 1;
    
    imwrite(rgb_gradient, fullfile(images_dir, 'color_gradient.png'));
    imwrite(hsv_img, fullfile(images_dir, 'color_wheel_hsv.png'));
    imwrite(primary_colors, fullfile(images_dir, 'primary_colors.png'));
end

function generate_morphology_test_images(images_dir)
    fprintf('  Generating morphology test images...\n');
    
    img_size = 400;
    binary_img = zeros(img_size, img_size);
    
    [x, y] = meshgrid(1:img_size, 1:img_size);
    
    # Various shapes for morphological operations
    
    # Large circle
    circle1 = (x - 100).^2 + (y - 100).^2 <= 40^2;
    binary_img = binary_img | circle1;
    
    # Small connected circles
    circle2 = (x - 200).^2 + (y - 80).^2 <= 15^2;
    circle3 = (x - 220).^2 + (y - 100).^2 <= 15^2;
    circle4 = (x - 200).^2 + (y - 120).^2 <= 15^2;
    binary_img = binary_img | circle2 | circle3 | circle4;
    
    # Rectangle with hole
    rect = (x >= 280 & x <= 350) & (y >= 50 & y <= 120);
    hole = (x >= 300 & x <= 330) & (y >= 70 & y <= 100);
    binary_img = binary_img | (rect & ~hole);
    
    # Thin connecting lines
    line1 = (abs(x - y + 50) <= 2) & (x >= 50) & (x <= 150);
    line2 = (abs(x + y - 300) <= 2) & (x >= 150) & (x <= 200);
    binary_img = binary_img | line1 | line2;
    
    # L-shaped structure
    l_shape_v = (x >= 50 & x <= 70) & (y >= 200 & y <= 300);
    l_shape_h = (x >= 50 & x <= 120) & (y >= 280 & y <= 300);
    binary_img = binary_img | l_shape_v | l_shape_h;
    
    # Cross shape
    cross_v = (x >= 195 & x <= 205) & (y >= 200 & y <= 280);
    cross_h = (x >= 160 & x <= 240) & (y >= 235 & y <= 245);
    binary_img = binary_img | cross_v | cross_h;
    
    # Noisy small objects (salt noise)
    noise_density = 0.005;
    salt_noise = rand(img_size, img_size) < noise_density;
    binary_img = binary_img | salt_noise;
    
    imwrite(double(binary_img), fullfile(images_dir, 'binary_shapes.png'));
    
    # Create grayscale version with different intensities
    gray_shapes = zeros(img_size, img_size);
    gray_shapes(circle1) = 0.8;
    gray_shapes(circle2 | circle3 | circle4) = 0.6;
    gray_shapes(rect & ~hole) = 1.0;
    gray_shapes(line1 | line2) = 0.4;
    gray_shapes(l_shape_v | l_shape_h) = 0.9;
    gray_shapes(cross_v | cross_h) = 0.7;
    gray_shapes(salt_noise) = 0.3;
    
    imwrite(gray_shapes, fullfile(images_dir, 'grayscale_shapes.png'));
end