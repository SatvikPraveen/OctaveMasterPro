% File location: OctaveMasterPro/datasets/create_sample_images.m
% Script to generate sample images for image processing learning

fprintf('Creating sample images for OctaveMasterPro...\n');

% Create directories if they don't exist
if ~exist('images', 'dir')
    mkdir('images');
end
if ~exist('images/samples', 'dir')
    mkdir('images/samples');
end
if ~exist('images/batch', 'dir')
    mkdir('images/batch');
end
if ~exist('images/medical', 'dir')
    mkdir('images/medical');
end
if ~exist('images/test', 'dir')
    mkdir('images/test');
end

% Image dimensions
width = 256;
height = 256;

%% Generate Test Pattern
fprintf('Creating test pattern...\n');
[X, Y] = meshgrid(1:width, 1:height);

% Create a complex test pattern with multiple features
test_pattern = zeros(height, width, 3);

% Checkerboard pattern
checker_size = 32;
checker = mod(floor(X/checker_size) + floor(Y/checker_size), 2);

% Circular patterns
center_x = width/2;
center_y = height/2;
radius = sqrt((X - center_x).^2 + (Y - center_y).^2);

% Combine patterns
test_pattern(:,:,1) = checker;
test_pattern(:,:,2) = (sin(radius/10) + 1) / 2;
test_pattern(:,:,3) = (cos(radius/8) + 1) / 2;

% Add grid lines
grid_spacing = 32;
test_pattern(1:grid_spacing:end, :, :) = 1;
test_pattern(:, 1:grid_spacing:end, :) = 1;

imwrite(test_pattern, 'images/test/test_pattern.png');

%% Generate Noisy Image
fprintf('Creating noisy image...\n');
% Start with a simple geometric shape
base_image = zeros(height, width);
base_image(64:192, 64:192) = 1; % Square
base_image(96:160, 96:160) = 0; % Square hole

% Add circles
circle_mask = (X - 200).^2 + (Y - 200).^2 < 30^2;
base_image(circle_mask) = 0.7;

% Add noise
noise_level = 0.3;
noise = noise_level * randn(height, width);
noisy_image = base_image + noise;
noisy_image = max(0, min(1, noisy_image)); % Clamp to [0,1]

imwrite(noisy_image, 'images/test/noisy_image.png');

%% Generate Sample Images (Synthetic but Realistic)
fprintf('Creating sample images...\n');

for i = 1:4
    sample_img = zeros(height, width, 3);
    
    switch i
        case 1 % Gradient with shapes
            % Background gradient
            for c = 1:3
                sample_img(:,:,c) = (X + Y * c/3) / (2 * width);
            end
            
            % Add geometric shapes
            circle1 = (X - 100).^2 + (Y - 100).^2 < 40^2;
            circle2 = (X - 180).^2 + (Y - 150).^2 < 30^2;
            sample_img(circle1, :) = repmat([1, 0.5, 0.2], sum(circle1(:)), 1);
            sample_img(circle2, :) = repmat([0.2, 0.8, 0.9], sum(circle2(:)), 1);
            
        case 2 % Texture pattern
            % Create a wood-like texture
            for c = 1:3
                rings = sin(radius / 10) * 0.3 + 0.5;
                grain = sin(Y / 3) * 0.1;
                sample_img(:,:,c) = rings + grain + 0.4 * (c-1)/2;
            end
            
        case 3 % Edge detection target
            % Create various edges and lines
            sample_img(:,:,1) = 0.3;
            sample_img(:,:,2) = 0.5;
            sample_img(:,:,3) = 0.7;
            
            % Horizontal lines
            sample_img(50:55, :, :) = 1;
            sample_img(100:105, :, :) = 0;
            sample_img(150:155, :, :) = 0.8;
            
            % Vertical lines
            sample_img(:, 80:85, :) = 0.2;
            sample_img(:, 170:175, :) = 0.9;
            
            % Diagonal line
            for j = 1:width
                row = round(j * height / width);
                if row >= 1 && row <= height
                    sample_img(max(1,row-2):min(height,row+2), j, :) = 0.1;
                end
            end
            
        case 4 % Blob detection target
            % Random blob-like structures
            sample_img(:,:,:) = 0.2; % Dark background
            
            % Generate several blobs
            blob_centers = [80, 80; 120, 170; 200, 100; 180, 200];
            blob_sizes = [25, 35, 20, 30];
            
            for b = 1:size(blob_centers, 1)
                cx = blob_centers(b, 1);
                cy = blob_centers(b, 2);
                blob_radius = blob_sizes(b);
                
                blob_mask = (X - cx).^2 + (Y - cy).^2 < blob_radius^2;
                
                % Gaussian-like intensity
                distances = sqrt((X - cx).^2 + (Y - cy).^2);
                gaussian_intensity = exp(-distances.^2 / (2 * (blob_radius/2)^2));
                
                for c = 1:3
                    sample_img(:,:,c) = sample_img(:,:,c) + ...
                        gaussian_intensity * 0.8 * (1 - (c-1)*0.2);
                end
            end
    end
    
    % Clamp values
    sample_img = max(0, min(1, sample_img));
    
    filename = sprintf('images/samples/sample_%02d.jpg', i);
    imwrite(sample_img, filename, 'Quality', 85);
end

%% Generate Batch Processing Images
fprintf('Creating batch processing images...\n');

for i = 1:10
    % Create variations for batch processing
    batch_img = zeros(height, width, 3);
    
    % Base pattern with variations
    freq = 0.02 + i * 0.01;
    phase_shift = i * pi / 5;
    
    pattern = sin(freq * 2 * pi * X + phase_shift) .* cos(freq * 2 * pi * Y);
    
    % Different color schemes for each image
    hue = mod(i * 0.3, 1);
    
    for c = 1:3
        color_shift = (c - 1) * 2 * pi / 3;
        batch_img(:,:,c) = 0.5 + 0.3 * pattern * (0.5 + 0.5 * sin(hue * 2 * pi + color_shift));
    end
    
    % Add some geometric elements
    if mod(i, 3) == 1
        % Add circle
        circle_center = [width/2 + randn()*30, height/2 + randn()*30];
        circle_mask = (X - circle_center(1)).^2 + (Y - circle_center(2)).^2 < (20 + i*2)^2;
        batch_img(circle_mask, :) = repmat([1, 1, 0], sum(circle_mask(:)), 1);
    end
    
    batch_img = max(0, min(1, batch_img));
    filename = sprintf('images/batch/batch_%02d.jpg', i);
    imwrite(batch_img, filename, 'Quality', 80);
end

%% Generate Medical-Style Images
fprintf('Creating medical-style images...\n');

% X-ray style image
xray_img = zeros(height, width);

% Simulate bone structures
bone_img = zeros(height, width);

% Femur-like structure
for y = 50:200
    thickness = 10 + 8 * sin((y-50) * pi / 150);
    center_x = width/2 + 20 * sin((y-50) * 2 * pi / 150);
    x_start = max(1, round(center_x - thickness/2));
    x_end = min(width, round(center_x + thickness/2));
    bone_img(y, x_start:x_end) = 0.9;
end

% Add joint
joint_center = [width/2, 220];
joint_radius = 25;
joint_mask = (X - joint_center(1)).^2 + (Y - joint_center(2)).^2 < joint_radius^2;
bone_img(joint_mask) = 0.7;

% Soft tissue (lower intensity)
soft_tissue = 0.3 * randn(height, width);
soft_tissue = max(0, min(0.4, 0.2 + soft_tissue * 0.1));

xray_img = soft_tissue + bone_img;
xray_img = max(0, min(1, xray_img));

imwrite(xray_img, 'images/medical/xray_sample.jpg', 'Quality', 90);

% Microscopy-style images
for i = 1:5
    micro_img = zeros(height, width, 3);
    
    % Create cell-like structures
    num_cells = 5 + round(randn() * 2);
    
    for cell = 1:num_cells
        center = [rand() * width, rand() * height];
        cell_radius = 15 + rand() * 20;
        
        % Cell body
        cell_mask = (X - center(1)).^2 + (Y - center(2)).^2 < cell_radius^2;
        
        % Different staining for each image
        if i <= 2
            % Blue/purple staining
            micro_img(cell_mask, 1) = 0.3;
            micro_img(cell_mask, 2) = 0.2;
            micro_img(cell_mask, 3) = 0.8;
        else
            % Green fluorescence
            micro_img(cell_mask, 1) = 0.2;
            micro_img(cell_mask, 2) = 0.8;
            micro_img(cell_mask, 3) = 0.3;
        end
        
        % Nucleus
        nucleus_radius = cell_radius * 0.4;
        nucleus_mask = (X - center(1)).^2 + (Y - center(2)).^2 < nucleus_radius^2;
        micro_img(nucleus_mask, :) = 0.1;
    end
    
    % Background
    background_level = 0.9;
    for c = 1:3
        background_mask = micro_img(:,:,c) == 0;
        micro_img(background_mask, c) = background_level;
    end
    
    filename = sprintf('images/medical/microscopy_%02d.png', i);
    imwrite(micro_img, filename);
end

fprintf('Successfully created all sample images!\n');
fprintf('Generated:\n');
fprintf('  - Test pattern and noisy image\n');
fprintf('  - 4 sample images for basic processing\n');
fprintf('  - 10 batch processing images\n');
fprintf('  - 1 X-ray style medical image\n');
fprintf('  - 5 microscopy style images\n');
fprintf('\nAll images are ready for use in the learning modules.\n');