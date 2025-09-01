% Location: mini_projects/image_processing_basics/morphology_ops.m
% Mathematical Morphology Operations

function se = create_structuring_element(shape, size_param, varargin)
    % Create structuring elements for morphological operations
    
    switch lower(shape)
        case 'disk'
            radius = size_param;
            [x, y] = meshgrid(-radius:radius, -radius:radius);
            se = double((x.^2 + y.^2) <= radius^2);
            
        case 'square'
            side = size_param;
            se = ones(side, side);
            
        case 'cross'
            size_val = size_param;
            se = zeros(2*size_val+1, 2*size_val+1);
            se(size_val+1, :) = 1;
            se(:, size_val+1) = 1;
            
        case 'line'
            length_val = size_param;
            angle = 0;
            if length(varargin) >= 1, angle = varargin{1}; end
            
            if angle == 0
                se = ones(1, length_val);
            elseif angle == 90
                se = ones(length_val, 1);
            else
                angle_rad = angle * pi / 180;
                x_end = round(length_val * cos(angle_rad));
                y_end = round(length_val * sin(angle_rad));
                max_dim = max(abs(x_end), abs(y_end)) + 1;
                se = zeros(2*max_dim+1, 2*max_dim+1);
                center = max_dim + 1;
                for t = 0:0.1:1
                    x = round(center + t * x_end);
                    y = round(center + t * y_end);
                    if x >= 1 && x <= size(se, 2) && y >= 1 && y <= size(se, 1)
                        se(y, x) = 1;
                    end
                end
            end
            
        case 'diamond'
            radius = size_param;
            [x, y] = meshgrid(-radius:radius, -radius:radius);
            se = double((abs(x) + abs(y)) <= radius);
    end
end

function result = morphological_erosion(img, se)
    % Morphological erosion operation
    
    if max(img(:)) > 1, img = img > 0.5; end
    
    [h, w] = size(img);
    [se_h, se_w] = size(se);
    pad_h = floor(se_h / 2);
    pad_w = floor(se_w / 2);
    padded_img = padarray(img, [pad_h, pad_w], 0);
    
    result = zeros(h, w);
    
    for i = 1:h
        for j = 1:w
            window = padded_img(i:i+se_h-1, j:j+se_w-1);
            masked_window = window(se > 0);
            result(i, j) = min(masked_window);
        end
    end
end

function result = morphological_dilation(img, se)
    % Morphological dilation operation
    
    if max(img(:)) > 1, img = img > 0.5; end
    
    [h, w] = size(img);
    [se_h, se_w] = size(se);
    pad_h = floor(se_h / 2);
    pad_w = floor(se_w / 2);
    padded_img = padarray(img, [pad_h, pad_w], 0);
    
    result = zeros(h, w);
    
    for i = 1:h
        for j = 1:w
            window = padded_img(i:i+se_h-1, j:j+se_w-1);
            masked_window = window(se > 0);
            result(i, j) = max(masked_window);
        end
    end
end

function result = morphological_opening(img, se)
    % Morphological opening (erosion followed by dilation)
    
    eroded = morphological_erosion(img, se);
    result = morphological_dilation(eroded, se);
end

function result = morphological_closing(img, se)
    % Morphological closing (dilation followed by erosion)
    
    dilated = morphological_dilation(img, se);
    result = morphological_erosion(dilated, se);
end

function result = morphological_gradient(img, se)
    % Morphological gradient (dilation - erosion)
    
    dilated = morphological_dilation(img, se);
    eroded = morphological_erosion(img, se);
    result = dilated - eroded;
end

function result = top_hat_transform(img, se)
    % Top-hat transform (original - opening)
    
    opened = morphological_opening(img, se);
    result = img - opened;
end

function result = black_hat_transform(img, se)
    % Black-hat transform (closing - original)
    
    closed = morphological_closing(img, se);
    result = closed - img;
end

function demo_morphological_operations()
    % Demonstrate all morphological operations
    
    fprintf('\n--- Morphological Operations Demonstration ---\n');
    
    % Create binary test image
    img = create_binary_test_image();
    
    % Create different structuring elements
    se_disk = create_structuring_element('disk', 3);
    se_square = create_structuring_element('square', 5);
    se_cross = create_structuring_element('cross', 2);
    
    % Apply operations
    eroded = morphological_erosion(img, se_disk);
    dilated = morphological_dilation(img, se_disk);
    opened = morphological_opening(img, se_square);
    closed = morphological_closing(img, se_square);
    gradient = morphological_gradient(img, se_cross);
    tophat = top_hat_transform(img, se_disk);
    blackhat = black_hat_transform(img, se_disk);
    
    % Display results
    figure('Position', [50, 50, 1400, 1000]);
    
    subplot(3, 3, 1);
    imshow(img, []);
    title('Original Binary Image');
    
    subplot(3, 3, 2);
    imshow(eroded, []);
    title('Erosion (Disk r=3)');
    
    subplot(3, 3, 3);
    imshow(dilated, []);
    title('Dilation (Disk r=3)');
    
    subplot(3, 3, 4);
    imshow(opened, []);
    title('Opening (Square 5x5)');
    
    subplot(3, 3, 5);
    imshow(closed, []);
    title('Closing (Square 5x5)');
    
    subplot(3, 3, 6);
    imshow(gradient, []);
    title('Morphological Gradient');
    
    subplot(3, 3, 7);
    imshow(tophat, []);
    title('Top-hat Transform');
    
    subplot(3, 3, 8);
    imshow(blackhat, []);
    title('Black-hat Transform');
    
    subplot(3, 3, 9);
    % Show structuring elements
    subplot(3, 3, 9);
    se_display = zeros(15, 45);
    se_display(6:10, 2:6) = se_square;
    se_display(6:12, 15:21) = se_disk;
    se_display(6:10, 35:39) = se_cross;
    imshow(se_display, []);
    title('Structuring Elements');
    
    sgtitle('Mathematical Morphology Operations');
    
    fprintf('Morphological operations demonstration complete.\n');
end

function binary_img = create_binary_test_image()
    % Create binary test image with various shapes
    
    img = zeros(200, 200);
    
    % Add circles
    [x, y] = meshgrid(1:200, 1:200);
    img = img + double((x-50).^2 + (y-50).^2 < 20^2);
    img = img + double((x-150).^2 + (y-50).^2 < 15^2);
    img = img + double((x-50).^2 + (y-150).^2 < 25^2);
    
    % Add rectangles
    img(100:130, 100:140) = 1;
    img(160:180, 120:180) = 1;
    
    % Add line structures
    img(80:120, 75:77) = 1;
    img(140:142, 80:120) = 1;
    
    binary_img = img > 0;
end

function compare_structuring_elements(img)
    % Compare effects of different structuring elements
    
    fprintf('\n--- Structuring Element Comparison ---\n');
    
    % Create different structuring elements
    se_disk3 = create_structuring_element('disk', 3);
    se_disk5 = create_structuring_element('disk', 5);
    se_square3 = create_structuring_element('square', 7);
    se_cross3 = create_structuring_element('cross', 3);
    se_line = create_structuring_element('line', 7, 45);
    
    structuring_elements = {se_disk3, se_disk5, se_square3, se_cross3, se_line};
    se_names = {'Disk r=3', 'Disk r=5', 'Square 7x7', 'Cross r=3', 'Line 45°'};
    
    figure('Position', [100, 100, 1400, 1000]);
    
    % Show original
    subplot(4, 6, 1);
    imshow(img, []);
    title('Original');
    
    % Show structuring elements
    for i = 1:5
        subplot(4, 6, i+1);
        imshow(structuring_elements{i}, []);
        title(se_names{i});
    end
    
    % Show erosion results
    for i = 1:5
        subplot(4, 6, 6+i+1);
        eroded = morphological_erosion(img, structuring_elements{i});
        imshow(eroded, []);
        title(['Erosion: ' se_names{i}]);
    end
    
    % Show dilation results
    for i = 1:5
        subplot(4, 6, 12+i+1);
        dilated = morphological_dilation(img, structuring_elements{i});
        imshow(dilated, []);
        title(['Dilation: ' se_names{i}]);
    end
    
    % Show opening results
    for i = 1:5
        subplot(4, 6, 18+i+1);
        opened = morphological_opening(img, structuring_elements{i});
        imshow(opened, []);
        title(['Opening: ' se_names{i}]);
    end
    
    sgtitle('Structuring Element Effects Comparison');
    
    fprintf('Structuring element comparison complete.\n');
end

function cleaned_img = noise_removal_demo(img)
    % Demonstrate morphological noise removal
    
    fprintf('\n--- Morphological Noise Removal ---\n');
    
    % Convert to binary if needed
    if max(img(:)) > 1
        binary_img = img > 128;
    else
        binary_img = img > 0.5;
    end
    
    % Add salt and pepper noise
    noisy_img = binary_img;
    noise_density = 0.05;
    
    % Salt noise (random white pixels)
    salt_locations = rand(size(binary_img)) < noise_density/2;
    noisy_img(salt_locations) = 1;
    
    % Pepper noise (random black pixels)
    pepper_locations = rand(size(binary_img)) < noise_density/2;
    noisy_img(pepper_locations) = 0;
    
    % Apply morphological cleaning
    se_small = create_structuring_element('disk', 1);
    se_medium = create_structuring_element('disk', 2);
    
    % Method 1: Opening followed by closing
    method1 = morphological_opening(noisy_img, se_small);
    method1 = morphological_closing(method1, se_small);
    
    % Method 2: Median-like morphological filtering
    method2 = morphological_closing(noisy_img, se_small);
    method2 = morphological_opening(method2, se_small);
    
    % Method 3: Multiple iterations with small SE
    method3 = noisy_img;
    for iter = 1:2
        method3 = morphological_opening(method3, se_small);
        method3 = morphological_closing(method3, se_small);
    end
    
    % Display results
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    imshow(binary_img, []);
    title('Original Binary Image');
    
    subplot(2, 3, 2);
    imshow(noisy_img, []);
    title('Salt & Pepper Noise Added');
    
    subplot(2, 3, 3);
    imshow(method1, []);
    title('Method 1: Open→Close');
    
    subplot(2, 3, 4);
    imshow(method2, []);
    title('Method 2: Close→Open');
    
    subplot(2, 3, 5);
    imshow(method3, []);
    title('Method 3: Iterative');
    
    % Performance comparison
    subplot(2, 3, 6);
    error1 = sum(sum(abs(binary_img - method1)));
    error2 = sum(sum(abs(binary_img - method2)));
    error3 = sum(sum(abs(binary_img - method3)));
    
    methods = {'Open→Close', 'Close→Open', 'Iterative'};
    errors = [error1, error2, error3];
    
    bar(errors);
    set(gca, 'XTickLabel', methods);
    title('Reconstruction Error');
    ylabel('Total Pixel Differences');
    grid on;
    
    sgtitle('Morphological Noise Removal Comparison');
    
    cleaned_img = method1; % Return best result
    
    fprintf('Noise removal demonstration complete.\n');
    fprintf('Reconstruction errors: %.0f, %.0f, %.0f\n', error1, error2, error3);
end

function boundary = extract_boundary(img, se)
    % Extract object boundaries using morphological operations
    
    if max(img(:)) > 1, img = img > 0.5; end
    
    % Boundary = Original - Erosion
    eroded = morphological_erosion(img, se);
    boundary = img - eroded;
end

function skeleton = morphological_skeletonization(img)
    % Simple skeletonization using iterative erosion
    
    if max(img(:)) > 1, img = img > 0.5; end
    
    se = create_structuring_element('cross', 1);
    skeleton = zeros(size(img));
    current = img;
    
    while sum(current(:)) > 0
        eroded = morphological_erosion(current, se);
        opened = morphological_opening(eroded, se);
        skeleton = skeleton | (eroded - opened);
        current = eroded;
    end
end

function demo_advanced_morphology()
    % Demonstrate advanced morphological operations
    
    fprintf('\n--- Advanced Morphology Demonstration ---\n');
    
    % Create complex test image
    img = create_complex_binary_image();
    
    % Apply advanced operations
    se = create_structuring_element('disk', 2);
    
    boundary = extract_boundary(img, se);
    skeleton = morphological_skeletonization(img);
    gradient = morphological_gradient(img, se);
    tophat = top_hat_transform(img, create_structuring_element('disk', 5));
    
    % Display results
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    imshow(img, []);
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(boundary, []);
    title('Boundary Extraction');
    
    subplot(2, 3, 3);
    imshow(skeleton, []);
    title('Morphological Skeleton');
    
    subplot(2, 3, 4);
    imshow(gradient, []);
    title('Morphological Gradient');
    
    subplot(2, 3, 5);
    imshow(tophat, []);
    title('Top-hat Transform');
    
    subplot(2, 3, 6);
    % Overlay boundary on original
    overlay = cat(3, img, img + boundary, img);
    imshow(overlay, []);
    title('Boundary Overlay');
    
    sgtitle('Advanced Morphological Operations');
    
    fprintf('Advanced morphology demonstration complete.\n');
end

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