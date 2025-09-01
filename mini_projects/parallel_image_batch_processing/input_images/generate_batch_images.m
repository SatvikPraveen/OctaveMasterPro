% Location: mini_projects/parallel_image_batch_processing/input_images/generate_batch_images.m
% Generate Batch Images for Parallel Processing Testing

function generate_batch_images()
    fprintf('Generating batch processing test images...\n');
    
    images_dir = fileparts(mfilename('fullpath'));
    if ~exist(images_dir, 'dir')
        mkdir(images_dir);
    end
    
    # Generate different categories for performance testing
    generate_simple_images(images_dir, 1, 15);      % batch_001 to batch_015
    generate_complex_images(images_dir, 16, 30);    % batch_016 to batch_030  
    generate_large_images(images_dir, 31, 40);      % batch_031 to batch_040
    generate_noisy_images(images_dir, 41, 50);      % batch_041 to batch_050
    
    fprintf('Batch images generation complete!\n');
    fprintf('Generated 50 test images in: %s\n', images_dir);
end

function generate_simple_images(images_dir, start_idx, end_idx)
    fprintf('  Generating simple images (%d-%d)...\n', start_idx, end_idx);
    
    for i = start_idx:end_idx
        img_size = 200;
        img = zeros(img_size, img_size, 3);
        [x, y] = meshgrid(1:img_size, 1:img_size);
        
        # Simple geometric pattern (fast to process)
        pattern_type = mod(i, 4) + 1;
        
        switch pattern_type
            case 1  # Single circle
                center = [img_size/2, img_size/2];
                radius = 30 + 20 * rand();
                circle_mask = (x - center(1)).^2 + (y - center(2)).^2 <= radius^2;
                
                color = rand(1, 3);
                for c = 1:3
                    img(:,:,c) = double(circle_mask) * color(c);
                end
                
            case 2  # Rectangle
                rect_size = [40 + 20*rand(), 60 + 20*rand()];
                rect_pos = [50 + 50*rand(), 50 + 50*rand()];
                
                rect_mask = (x >= rect_pos(1) & x <= rect_pos(1) + rect_size(1)) & ...
                           (y >= rect_pos(2) & y <= rect_pos(2) + rect_size(2));
                
                color = rand(1, 3);
                for c = 1:3
                    img(:,:,c) = double(rect_mask) * color(c);
                end
                
            case 3  # Gradient
                img(:,:,1) = x / img_size;
                img(:,:,2) = y / img_size;  
                img(:,:,3) = (x + y) / (2 * img_size);
                
            case 4  # Stripes
                stripe_width = 10 + 5 * rand();
                stripe_mask = mod(x, stripe_width*2) < stripe_width;
                
                color = rand(1, 3);
                for c = 1:3
                    img(:,:,c) = double(stripe_mask) * color(c);
                end
        end
        
        # Add small amount of noise
        img = img + 0.02 * randn(size(img));
        img = max(0, min(1, img));
        
        filename = sprintf('batch_%03d.png', i);
        imwrite(img, fullfile(images_dir, filename));
    end
end

function generate_complex_images(images_dir, start_idx, end_idx)
    fprintf('  Generating complex images (%d-%d)...\n', start_idx, end_idx);
    
    for i = start_idx:end_idx
        img_size = 300;
        img = zeros(img_size, img_size, 3);
        [x, y] = meshgrid(1:img_size, 1:img_size);
        
        # Complex patterns (moderate processing time)
        pattern_type = mod(i, 3) + 1;
        
        switch pattern_type
            case 1  # Multiple overlapping shapes
                num_shapes = 5 + round(5 * rand());
                
                for shape = 1:num_shapes
                    center = [img_size * rand(), img_size * rand()];
                    radius = 15 + 25 * rand();
                    
                    if mod(shape, 2) == 1  # Circle
                        mask = (x - center(1)).^2 + (y - center(2)).^2 <= radius^2;
                    else  # Rectangle
                        size_w = radius;
                        size_h = radius * (0.5 + 0.5*rand());
                        mask = (abs(x - center(1)) <= size_w/2) & (abs(y - center(2)) <= size_h/2);
                    end
                    
                    color = rand(1, 3) * 0.8;
                    alpha = 0.3 + 0.4 * rand();  # Transparency
                    
                    for c = 1:3
                        img(:,:,c) = (1-alpha) * img(:,:,c) + alpha * color(c) * double(mask);
                    end
                end
                
            case 2  # Frequency patterns
                freq_x = 2 + 8 * rand();
                freq_y = 2 + 8 * rand();
                phase_x = 2*pi*rand();
                phase_y = 2*pi*rand();
                
                pattern = sin(2*pi*freq_x*x/img_size + phase_x) .* cos(2*pi*freq_y*y/img_size + phase_y);
                pattern = (pattern + 1) / 2;  # Normalize to [0,1]
                
                # Apply to different channels with phase shifts
                img(:,:,1) = pattern;
                img(:,:,2) = circshift(pattern, [round(img_size/8), 0]);
                img(:,:,3) = circshift(pattern, [0, round(img_size/8)]);
                
            case 3  # Voronoi-like cellular pattern
                num_seeds = 8 + round(12 * rand());
                seeds = [img_size * rand(num_seeds, 1), img_size * rand(num_seeds, 1)];
                seed_colors = rand(num_seeds, 3);
                
                for px = 1:img_size
                    for py = 1:img_size
                        distances = sqrt((seeds(:,1) - px).^2 + (seeds(:,2) - py).^2);
                        [~, closest_seed] = min(distances);
                        img(py, px, :) = seed_colors(closest_seed, :);
                    end
                end
                
                # Add some smoothing for visual appeal
                for c = 1:3
                    img(:,:,c) = conv2(img(:,:,c), ones(3)/9, 'same');
                end
        end
        
        # Add moderate noise
        img = img + 0.05 * randn(size(img));
        img = max(0, min(1, img));
        
        filename = sprintf('batch_%03d.png', i);
        imwrite(img, fullfile(images_dir, filename));
    end
end

function generate_large_images(images_dir, start_idx, end_idx)
    fprintf('  Generating large images (%d-%d)...\n', start_idx, end_idx);
    
    for i = start_idx:end_idx
        img_size = 500 + round(300 * rand());  # 500-800 pixels
        img = zeros(img_size, img_size, 3);
        [x, y] = meshgrid(1:img_size, 1:img_size);
        
        # Large images (memory intensive)
        pattern_type = mod(i, 2) + 1;
        
        switch pattern_type
            case 1  # High resolution texture
                # Multi-scale noise pattern
                scales = [32, 64, 128, 256];
                weights = [0.4, 0.3, 0.2, 0.1];
                
                for c = 1:3
                    channel = zeros(img_size, img_size);
                    
                    for s = 1:length(scales)
                        scale = scales(s);
                        weight = weights(s);
                        
                        # Create noise at this scale
                        noise_size = ceil(img_size / scale);
                        noise = randn(noise_size, noise_size);
                        
                        # Upscale to full size
                        [noise_x, noise_y] = meshgrid(linspace(1, noise_size, img_size));
                        scaled_noise = interp2(noise, noise_x, noise_y, 'linear', 0);
                        
                        channel = channel + weight * scaled_noise;
                    end
                    
                    img(:,:,c) = 0.5 + 0.3 * channel / max(abs(channel(:)));
                end
                
            case 2  # Complex geometric pattern
                # Create fractal-like recursive pattern
                img = create_fractal_pattern(img_size);
        end
        
        # Large images get less noise to maintain quality
        img = img + 0.01 * randn(size(img));
        img = max(0, min(1, img));
        
        filename = sprintf('batch_%03d.png', i);
        imwrite(img, fullfile(images_dir, filename));
    end
end

function generate_noisy_images(images_dir, start_idx, end_idx)
    fprintf('  Generating noisy images (%d-%d)...\n', start_idx, end_idx);
    
    for i = start_idx:end_idx
        img_size = 250;
        [x, y] = meshgrid(1:img_size, 1:img_size);
        
        # Start with clean pattern
        clean_img = zeros(img_size, img_size, 3);
        
        # Base pattern
        base_pattern = sin(2*pi*x/30) .* cos(2*pi*y/40) + 0.5 * sin(2*pi*x/60);
        base_pattern = (base_pattern + 1.5) / 3; # Normalize
        
        for c = 1:3
            clean_img(:,:,c) = circshift(base_pattern, [0, c*10]);
        end
        
        # Add various types of noise (computationally intensive to clean)
        noise_type = mod(i, 4) + 1;
        
        switch noise_type
            case 1  # Gaussian noise
                noise_level = 0.1 + 0.1 * rand();
                noisy_img = clean_img + noise_level * randn(size(clean_img));
                
            case 2  # Salt and pepper
                sp_density = 0.02 + 0.03 * rand();
                noisy_img = clean_img;
                
                salt_mask = rand(size(clean_img)) < sp_density/2;
                pepper_mask = rand(size(clean_img)) < sp_density/2;
                
                noisy_img(salt_mask) = 1;
                noisy_img(pepper_mask) = 0;
                
            case 3  # Speckle (multiplicative) noise
                speckle_variance = 0.1 + 0.1 * rand();
                noisy_img = clean_img .* (1 + speckle_variance * randn(size(clean_img)));
                
            case 4  # Mixed noise
                # Combination of different noise types
                gaussian_noise = 0.05 * randn(size(clean_img));
                
                sp_density = 0.01;
                salt_mask = rand(size(clean_img)) < sp_density/2;
                pepper_mask = rand(size(clean_img)) < sp_density/2;
                
                noisy_img = clean_img + gaussian_noise;
                noisy_img(salt_mask) = 1;
                noisy_img(pepper_mask) = 0;
                
                # Add some blur to make denoising more challenging
                blur_kernel = ones(3,3) / 9;
                for c = 1:3
                    noisy_img(:,:,c) = conv2(noisy_img(:,:,c), blur_kernel, 'same');
                end
        end
        
        noisy_img = max(0, min(1, noisy_img));
        
        filename = sprintf('batch_%03d.png', i);
        imwrite(noisy_img, fullfile(images_dir, filename));
    end
end

function fractal_img = create_fractal_pattern(img_size)
    # Create a simple fractal-like pattern
    
    fractal_img = zeros(img_size, img_size, 3);
    [x, y] = meshgrid(linspace(-2, 2, img_size), linspace(-2, 2, img_size));
    
    # Julia set-inspired pattern (simplified)
    c = -0.7 + 0.27015i;  # Julia set parameter
    z = x + 1i*y;
    
    max_iter = 50;
    escape_count = zeros(img_size, img_size);
    
    for iter = 1:max_iter
        # Julia set iteration: z = z^2 + c
        z = z.^2 + c;
        
        # Check for escape (magnitude > 2)
        escaped = abs(z) > 2;
        escape_count(escaped & escape_count == 0) = iter;
    end
    
    # Normalize escape count
    escape_count(escape_count == 0) = max_iter;
    escape_normalized = escape_count / max_iter;
    
    # Create colorful fractal
    fractal_img(:,:,1) = sin(2*pi*escape_normalized);
    fractal_img(:,:,2) = sin(2*pi*escape_normalized + 2*pi/3);
    fractal_img(:,:,3) = sin(2*pi*escape_normalized + 4*pi/3);
    
    # Normalize to [0,1]
    fractal_img = (fractal_img + 1) / 2;
end

function create_image_info_file(images_dir)
    # Create an info file describing the generated images
    
    info_filename = fullfile(images_dir, 'batch_info.txt');
    fid = fopen(info_filename, 'w');
    
    fprintf(fid, 'Batch Image Information\n');
    fprintf(fid, '=======================\n\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'Image Categories:\n');
    fprintf(fid, '-----------------\n');
    fprintf(fid, 'batch_001 - batch_015: Simple geometric patterns (200x200)\n');
    fprintf(fid, '  - Fast processing, low memory usage\n');
    fprintf(fid, '  - Single shapes, gradients, simple stripes\n\n');
    
    fprintf(fid, 'batch_016 - batch_030: Complex patterns (300x300)\n');
    fprintf(fid, '  - Moderate processing time\n');
    fprintf(fid, '  - Overlapping shapes, frequency patterns, cellular patterns\n\n');
    
    fprintf(fid, 'batch_031 - batch_040: Large images (500-800x500-800)\n');
    fprintf(fid, '  - Memory intensive\n');
    fprintf(fid, '  - High resolution textures, fractal patterns\n\n');
    
    fprintf(fid, 'batch_041 - batch_050: Noisy images (250x250)\n');
    fprintf(fid, '  - Computationally intensive to process\n');
    fprintf(fid, '  - Various noise types: Gaussian, salt&pepper, speckle, mixed\n\n');
    
    fprintf(fid, 'Usage:\n');
    fprintf(fid, '------\n');
    fprintf(fid, 'These images are designed for benchmarking parallel processing:\n');
    fprintf(fid, '- Simple images: Test basic parallel overhead\n');
    fprintf(fid, '- Complex images: Test computational scalability\n');
    fprintf(fid, '- Large images: Test memory management\n');
    fprintf(fid, '- Noisy images: Test algorithm efficiency\n\n');
    
    fclose(fid);
    
    fprintf('  Created batch_info.txt with image descriptions\n');
end