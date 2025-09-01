% Location: mini_projects/image_processing_basics/image_loader.m
% Image Loading and Preprocessing Utilities

function img = load_image(filename, varargin)
    % Load and preprocess images with various options
    %
    % Inputs:
    %   filename - path to image file
    %   varargin - optional parameters:
    %     'resize' - [height, width] to resize image
    %     'grayscale' - convert to grayscale (true/false)
    %     'normalize' - normalize pixel values to [0,1] (true/false)
    %     'enhance' - apply basic enhancement (true/false)
    %
    % Output:
    %   img - processed image matrix
    
    % Default parameters
    resize_dims = [];
    convert_grayscale = false;
    normalize_img = true;
    enhance_img = false;
    
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'resize'
                resize_dims = varargin{i+1};
            case 'grayscale'
                convert_grayscale = varargin{i+1};
            case 'normalize'
                normalize_img = varargin{i+1};
            case 'enhance'
                enhance_img = varargin{i+1};
        end
    end
    
    try
        % Load image
        if exist(filename, 'file')
            img = imread(filename);
        else
            % Create a synthetic test image if file doesn't exist
            fprintf('File not found. Creating synthetic test image...\n');
            img = create_test_image();
        end
        
        % Convert to double precision
        img = double(img);
        
        % Convert to grayscale if requested
        if convert_grayscale && size(img, 3) == 3
            img = rgb2gray_custom(img);
        end
        
        % Resize if requested
        if ~isempty(resize_dims)
            img = resize_image(img, resize_dims);
        end
        
        % Normalize pixel values
        if normalize_img
            if max(img(:)) > 1
                img = img / 255;
            end
        end
        
        % Apply basic enhancement
        if enhance_img
            img = enhance_image(img);
        end
        
    catch err
        fprintf('Error loading image: %s\n', err.message);
        fprintf('Creating synthetic test image instead...\n');
        img = create_test_image();
        
        if normalize_img
            img = img / 255;
        end
    end
end

function gray_img = rgb2gray_custom(rgb_img)
    % Convert RGB to grayscale using standard weights
    
    if size(rgb_img, 3) ~= 3
        gray_img = rgb_img;
        return;
    end
    
    % Standard RGB to grayscale conversion weights
    weights = [0.299, 0.587, 0.114];
    gray_img = rgb_img(:,:,1) * weights(1) + ...
               rgb_img(:,:,2) * weights(2) + ...
               rgb_img(:,:,3) * weights(3);
end

function resized_img = resize_image(img, new_dims)
    % Simple bilinear interpolation resize
    
    [old_h, old_w, channels] = size(img);
    new_h = new_dims(1);
    new_w = new_dims(2);
    
    % Create coordinate grids
    [X_old, Y_old] = meshgrid(1:old_w, 1:old_h);
    [X_new, Y_new] = meshgrid(linspace(1, old_w, new_w), linspace(1, old_h, new_h));
    
    % Interpolate each channel
    if channels == 1
        resized_img = interp2(X_old, Y_old, img, X_new, Y_new, 'linear', 0);
    else
        resized_img = zeros(new_h, new_w, channels);
        for c = 1:channels
            resized_img(:,:,c) = interp2(X_old, Y_old, img(:,:,c), X_new, Y_new, 'linear', 0);
        end
    end
end

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

function test_img = create_test_image(type)
    % Create synthetic test images for demonstration
    
    if nargin < 1
        type = 'mixed';
    end
    
    switch lower(type)
        case 'circles'
            % Image with circles of different sizes
            [x, y] = meshgrid(-100:100, -100:100);
            test_img = zeros(201, 201);
            
            % Add circles
            test_img = test_img + double((x.^2 + y.^2) < 30^2) * 255;
            test_img = test_img + double(((x-50).^2 + (y-30).^2) < 20^2) * 180;
            test_img = test_img + double(((x+40).^2 + (y+40).^2) < 15^2) * 120;
            
        case 'stripes'
            % Vertical and horizontal stripes
            test_img = zeros(200, 200);
            test_img(:, 20:40) = 255;
            test_img(:, 80:100) = 200;
            test_img(:, 140:160) = 150;
            test_img(60:80, :) = test_img(60:80, :) + 100;
            test_img = min(test_img, 255);
            
        case 'mixed'
            % Complex test pattern
            [x, y] = meshgrid(1:256, 1:256);
            test_img = zeros(256, 256);
            
            % Geometric shapes
            test_img = test_img + double((x-64).^2 + (y-64).^2 < 30^2) * 200;
            test_img = test_img + double(abs(x-192) < 20 & abs(y-64) < 20) * 180;
            test_img = test_img + double(abs(x-64-y+192) < 10) * 160;
            
            % Add noise and texture
            test_img = test_img + 30 * randn(256, 256);
            test_img = max(0, min(255, test_img));
            
        case 'gradient'
            % Smooth gradient for testing filters
            [x, y] = meshgrid(1:200, 1:200);
            test_img = x + y;
            test_img = 255 * test_img / max(test_img(:));
            
        otherwise
            % Default: simple geometric pattern
            test_img = create_test_image('mixed');
    end
    
    test_img = uint8(test_img);
end

function batch_img = load_image_batch(image_list, varargin)
    % Load multiple images as a batch
    %
    % Inputs:
    %   image_list - cell array of image filenames
    %   varargin - same options as load_image
    %
    % Output:
    %   batch_img - cell array of loaded images
    
    num_images = length(image_list);
    batch_img = cell(num_images, 1);
    
    fprintf('Loading %d images...\n', num_images);
    
    for i = 1:num_images
        fprintf('  Loading image %d/%d: %s\n', i, num_images, image_list{i});
        batch_img{i} = load_image(image_list{i}, varargin{:});
    end
    
    fprintf('Batch loading complete.\n');
end

function save_image(img, filename, varargin)
    % Save image with optional format conversion
    %
    % Inputs:
    %   img - image matrix
    %   filename - output filename
    %   varargin - optional parameters:
    %     'quality' - JPEG quality (1-100)
    %     'format' - force specific format
    
    % Default parameters
    quality = 95;
    format = '';
    
    % Parse optional arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'quality'
                quality = varargin{i+1};
            case 'format'
                format = varargin{i+1};
        end
    end
    
    % Ensure proper data type and range
    if max(img(:)) <= 1
        img = img * 255;
    end
    img = uint8(img);
    
    try
        if ~isempty(format)
            imwrite(img, filename, format, 'Quality', quality);
        else
            imwrite(img, filename, 'Quality', quality);
        end
        fprintf('Image saved: %s\n', filename);
    catch err
        fprintf('Error saving image: %s\n', err.message);
    end
end

function info = get_image_info(img)
    % Get comprehensive information about an image
    
    [h, w, c] = size(img);
    
    info.height = h;
    info.width = w;
    info.channels = c;
    info.total_pixels = h * w;
    info.data_type = class(img);
    
    if max(img(:)) <= 1
        info.range = [0, 1];
        info.normalized = true;
    else
        info.range = [0, 255];
        info.normalized = false;
    end
    
    info.min_value = min(img(:));
    info.max_value = max(img(:));
    info.mean_value = mean(img(:));
    info.std_value = std(img(:));
    
    if c == 1
        info.type = 'grayscale';
    else
        info.type = 'color';
    end
    
    % Display information
    fprintf('\nImage Information:\n');
    fprintf('------------------\n');
    fprintf('Dimensions: %d x %d x %d\n', h, w, c);
    fprintf('Type: %s\n', info.type);
    fprintf('Data type: %s\n', info.data_type);
    fprintf('Value range: [%.3f, %.3f]\n', info.min_value, info.max_value);
    fprintf('Mean: %.3f, Std: %.3f\n', info.mean_value, info.std_value);
    fprintf('Total pixels: %d\n', info.total_pixels);
end

function demo_image_loading()
    % Demonstrate image loading capabilities
    
    fprintf('\n--- Image Loading Demonstration ---\n');
    
    % Create various test images
    test_images = cell(4, 1);
    test_images{1} = create_test_image('circles');
    test_images{2} = create_test_image('stripes');
    test_images{3} = create_test_image('mixed');
    test_images{4} = create_test_image('gradient');
    
    titles = {'Circles', 'Stripes', 'Mixed Pattern', 'Gradient'};
    
    figure('Position', [100, 100, 1200, 800]);
    
    for i = 1:4
        subplot(2, 2, i);
        imshow(test_images{i}, []);
        title(titles{i});
        
        % Show image info
        get_image_info(double(test_images{i}));
    end
    
    sgtitle('Test Image Gallery');
    
    fprintf('Image loading demonstration complete.\n');
end