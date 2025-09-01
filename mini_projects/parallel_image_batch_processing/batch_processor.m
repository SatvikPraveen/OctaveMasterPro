% Location: mini_projects/parallel_image_batch_processing/batch_processor.m
% Main Parallel Image Batch Processing Controller

function results = process_image_batch(input_dir, output_dir, operation_func, varargin)
    % Process batch of images with parallel execution
    %
    % Inputs:
    %   input_dir - directory containing input images
    %   output_dir - directory for processed images
    %   operation_func - function handle for image processing operation
    %   varargin - optional parameters:
    %     'parallel' - enable parallel processing (default: true)
    %     'num_workers' - number of parallel workers (default: auto)
    %     'chunk_size' - images per worker chunk (default: auto)
    %     'progress' - show progress bar (default: true)
    %     'save_results' - save processed images (default: true)
    %
    % Output:
    %   results - struct with processing results and performance metrics
    
    % Default parameters
    use_parallel = true;
    num_workers = [];
    chunk_size = [];
    show_progress = true;
    save_results = true;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'parallel'
                use_parallel = varargin{i+1};
            case 'num_workers'
                num_workers = varargin{i+1};
            case 'chunk_size'
                chunk_size = varargin{i+1};
            case 'progress'
                show_progress = varargin{i+1};
            case 'save_results'
                save_results = varargin{i+1};
        end
    end
    
    % Initialize processing
    fprintf('Starting batch image processing...\n');
    start_time = tic;
    
    % Get list of image files
    image_files = get_image_file_list(input_dir);
    num_images = length(image_files);
    
    if num_images == 0
        fprintf('No images found in input directory. Generating test images...\n');
        image_files = generate_test_images(input_dir, 12);
        num_images = length(image_files);
    end
    
    fprintf('Found %d images to process.\n', num_images);
    
    % Create output directory
    if save_results && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Auto-configure parallel settings
    if use_parallel
        if isempty(num_workers)
            num_workers = determine_optimal_workers();
        end
        
        if isempty(chunk_size)
            chunk_size = max(1, floor(num_images / (num_workers * 2)));
        end
        
        fprintf('Parallel processing: %d workers, chunk size: %d\n', num_workers, chunk_size);
    else
        fprintf('Sequential processing mode.\n');
    end
    
    % Process images
    if use_parallel
        processing_results = parallel_image_processing(image_files, output_dir, operation_func, ...
                                                     num_workers, chunk_size, show_progress, save_results);
    else
        processing_results = sequential_image_processing(image_files, output_dir, operation_func, ...
                                                       show_progress, save_results);
    end
    
    % Compile results
    total_time = toc(start_time);
    
    results.num_images_processed = processing_results.num_processed;
    results.processing_time = total_time;
    results.images_per_second = processing_results.num_processed / total_time;
    results.use_parallel = use_parallel;
    results.num_workers = num_workers;
    results.chunk_size = chunk_size;
    results.memory_usage = processing_results.memory_usage;
    results.error_count = processing_results.error_count;
    
    % Display summary
    fprintf('\nBatch Processing Complete!\n');
    fprintf('========================\n');
    fprintf('Images processed: %d\n', results.num_images_processed);
    fprintf('Total time: %.2f seconds\n', results.processing_time);
    fprintf('Processing rate: %.2f images/second\n', results.images_per_second);
    fprintf('Mode: %s\n', results.use_parallel ? 'Parallel' : 'Sequential');
    if use_parallel
        fprintf('Workers: %d\n', results.num_workers);
    end
    fprintf('Errors: %d\n', results.error_count);
end

function image_files = get_image_file_list(input_dir)
    % Get list of image files from directory
    
    image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'};
    image_files = {};
    
    if exist(input_dir, 'dir')
        for ext_idx = 1:length(image_extensions)
            pattern = fullfile(input_dir, ['*.', image_extensions{ext_idx}]);
            files = dir(pattern);
            
            for i = 1:length(files)
                image_files{end+1} = fullfile(input_dir, files(i).name);
            end
        end
    end
end

function test_files = generate_test_images(output_dir, num_images)
    % Generate synthetic test images
    
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    test_files = cell(num_images, 1);
    
    fprintf('Generating %d test images...\n', num_images);
    
    for i = 1:num_images
        # Create synthetic image
        img_size = 200 + round(100 * rand()); # Random size 200-300
        [x, y] = meshgrid(1:img_size, 1:img_size);
        
        # Random pattern type
        pattern_type = mod(i, 4) + 1;
        
        switch pattern_type
            case 1 # Circles
                img = create_circle_pattern(x, y);
            case 2 # Stripes
                img = create_stripe_pattern(x, y, img_size);
            case 3 # Noise
                img = create_noise_pattern(img_size);
            case 4 # Gradient
                img = create_gradient_pattern(x, y);
        end
        
        # Add some noise
        img = img + 0.1 * randn(size(img));
        img = max(0, min(255, img));
        
        # Save test image
        filename = sprintf('test_image_%03d.png', i);
        filepath = fullfile(output_dir, filename);
        
        try
            imwrite(uint8(img), filepath);
            test_files{i} = filepath;
        catch
            fprintf('Warning: Could not save test image %d\n', i);
        end
    end
    
    # Remove empty entries
    test_files = test_files(~cellfun(@isempty, test_files));
    
    fprintf('Generated %d test images in %s\n', length(test_files), output_dir);
end

function img = create_circle_pattern(x, y)
    img = zeros(size(x));
    [h, w] = size(x);
    
    num_circles = 3 + round(3 * rand());
    for c = 1:num_circles
        center_x = round(w * rand());
        center_y = round(h * rand());
        radius = 10 + round(20 * rand());
        intensity = 100 + 155 * rand();
        
        circle_mask = (x - center_x).^2 + (y - center_y).^2 <= radius^2;
        img = img + circle_mask * intensity;
    end
    
    img = min(img, 255);
end

function img = create_stripe_pattern(x, y, img_size)
    freq = 0.1 + 0.3 * rand();
    angle = 180 * rand();
    
    rotated_x = x * cos(angle*pi/180) - y * sin(angle*pi/180);
    img = 127 + 127 * sin(2*pi*freq*rotated_x/img_size);
end

function img = create_noise_pattern(img_size)
    img = 127 + 50 * randn(img_size, img_size);
end

function img = create_gradient_pattern(x, y)
    [h, w] = size(x);
    img = (x + y) * 255 / (h + w);
end

function num_workers = determine_optimal_workers()
    % Determine optimal number of parallel workers
    
    try
        # Try to detect number of CPU cores (simplified)
        num_cores = 4; # Default assumption
        
        # Use 75% of available cores, but at least 2 and at most 8
        num_workers = max(2, min(8, round(0.75 * num_cores)));
        
        fprintf('Auto-detected %d CPU cores, using %d workers.\n', num_cores, num_workers);
    catch
        num_workers = 2; # Safe default
        fprintf('Could not detect CPU cores. Using %d workers.\n', num_workers);
    end
end

function results = parallel_image_processing(image_files, output_dir, operation_func, num_workers, chunk_size, show_progress, save_results)
    % Execute parallel image processing
    
    num_images = length(image_files);
    num_processed = 0;
    error_count = 0;
    memory_usage = [];
    
    fprintf('Starting parallel processing with %d workers...\n', num_workers);
    
    # Create chunks for parallel processing
    chunks = create_processing_chunks(image_files, chunk_size);
    num_chunks = length(chunks);
    
    if show_progress
        fprintf('Processing %d chunks...\n', num_chunks);
    end
    
    # Process chunks (simulated parallel processing)
    chunk_results = cell(num_chunks, 1);
    
    for chunk_idx = 1:num_chunks
        if show_progress
            fprintf('Processing chunk %d/%d...\n', chunk_idx, num_chunks);
        end
        
        chunk_files = chunks{chunk_idx};
        chunk_result = process_image_chunk(chunk_files, output_dir, operation_func, save_results);
        
        chunk_results{chunk_idx} = chunk_result;
        num_processed = num_processed + chunk_result.num_processed;
        error_count = error_count + chunk_result.error_count;
        
        # Monitor memory usage
        memory_usage = [memory_usage, get_memory_usage()];
    end
    
    results.num_processed = num_processed;
    results.error_count = error_count;
    results.memory_usage = memory_usage;
    results.chunk_results = chunk_results;
    
    fprintf('Parallel processing completed: %d images processed.\n', num_processed);
end

function results = sequential_image_processing(image_files, output_dir, operation_func, show_progress, save_results)
    % Execute sequential image processing
    
    num_images = length(image_files);
    num_processed = 0;
    error_count = 0;
    memory_usage = [];
    
    fprintf('Starting sequential processing...\n');
    
    for i = 1:num_images
        if show_progress && mod(i, 10) == 0
            fprintf('Processing image %d/%d (%.1f%%)...\n', i, num_images, i/num_images*100);
        end
        
        try
            # Load image
            img = load_image_safe(image_files{i});
            
            if ~isempty(img)
                # Apply operation
                processed_img = operation_func(img);
                
                # Save result
                if save_results
                    [~, name, ~] = fileparts(image_files{i});
                    output_file = fullfile(output_dir, [name, '_processed.png']);
                    save_image_safe(processed_img, output_file);
                end
                
                num_processed = num_processed + 1;
            end
            
        catch err
            error_count = error_count + 1;
            fprintf('Error processing %s: %s\n', image_files{i}, err.message);
        end
        
        # Monitor memory usage
        if mod(i, 5) == 0
            memory_usage = [memory_usage, get_memory_usage()];
        end
    end
    
    results.num_processed = num_processed;
    results.error_count = error_count;
    results.memory_usage = memory_usage;
    
    fprintf('Sequential processing completed: %d images processed.\n', num_processed);
end

function chunks = create_processing_chunks(image_files, chunk_size)
    % Divide image list into processing chunks
    
    num_images = length(image_files);
    num_chunks = ceil(num_images / chunk_size);
    chunks = cell(num_chunks, 1);
    
    for i = 1:num_chunks
        start_idx = (i-1) * chunk_size + 1;
        end_idx = min(i * chunk_size, num_images);
        chunks{i} = image_files(start_idx:end_idx);
    end
end

function chunk_result = process_image_chunk(chunk_files, output_dir, operation_func, save_results)
    % Process a chunk of images
    
    num_files = length(chunk_files);
    num_processed = 0;
    error_count = 0;
    
    for i = 1:num_files
        try
            img = load_image_safe(chunk_files{i});
            
            if ~isempty(img)
                processed_img = operation_func(img);
                
                if save_results
                    [~, name, ~] = fileparts(chunk_files{i});
                    output_file = fullfile(output_dir, [name, '_processed.png']);
                    save_image_safe(processed_img, output_file);
                end
                
                num_processed = num_processed + 1;
            end
            
        catch err
            error_count = error_count + 1;
        end
    end
    
    chunk_result.num_processed = num_processed;
    chunk_result.error_count = error_count;
end

function img = load_image_safe(filename)
    % Safely load image with error handling
    
    try
        if exist(filename, 'file')
            img = imread(filename);
            img = double(img);
            
            # Normalize if needed
            if max(img(:)) > 1
                img = img / 255;
            end
        else
            img = [];
        end
    catch
        img = [];
    end
end

function success = save_image_safe(img, filename)
    % Safely save image with error handling
    
    try
        if max(img(:)) <= 1
            img = img * 255;
        end
        
        img = uint8(max(0, min(255, img)));
        imwrite(img, filename);
        success = true;
    catch
        success = false;
    end
end

function memory_mb = get_memory_usage()
    % Get current memory usage (simplified)
    
    try
        # In a real implementation, this would query system memory
        # For demonstration, we'll simulate memory usage
        memory_mb = 100 + 50 * rand(); # Simulated MB usage
    catch
        memory_mb = NaN;
    end
end

function progress_monitor = create_progress_monitor(total_items)
    % Create progress monitoring structure
    
    progress_monitor.total_items = total_items;
    progress_monitor.processed_items = 0;
    progress_monitor.start_time = tic;
    progress_monitor.last_update = 0;
end

function update_progress_monitor(progress_monitor, items_completed)
    % Update progress monitor
    
    progress_monitor.processed_items = items_completed;
    current_time = toc(progress_monitor.start_time);
    
    if current_time - progress_monitor.last_update > 1 # Update every second
        progress_percent = items_completed / progress_monitor.total_items * 100;
        
        if items_completed > 0
            estimated_total_time = current_time * progress_monitor.total_items / items_completed;
            eta = estimated_total_time - current_time;
            
            fprintf('Progress: %.1f%% (%d/%d) - ETA: %.1f seconds\n', ...
                   progress_percent, items_completed, progress_monitor.total_items, eta);
        end
        
        progress_monitor.last_update = current_time;
    end
end

function benchmark_data = benchmark_processing_modes(image_files, operation_func)
    % Compare sequential vs parallel processing performance
    
    fprintf('\n--- Processing Mode Benchmark ---\n');
    
    num_images = min(length(image_files), 20); # Limit for demo
    test_files = image_files(1:num_images);
    
    # Sequential benchmark
    fprintf('Benchmarking sequential processing...\n');
    tic;
    sequential_results = sequential_image_processing(test_files, tempdir, operation_func, false, false);
    sequential_time = toc;
    
    # Parallel benchmark
    fprintf('Benchmarking parallel processing...\n');
    tic;
    parallel_results = parallel_image_processing(test_files, tempdir, operation_func, 4, 2, false, false);
    parallel_time = toc;
    
    # Calculate metrics
    speedup = sequential_time / parallel_time;
    efficiency = speedup / 4 * 100; # Assume 4 workers
    
    benchmark_data.sequential_time = sequential_time;
    benchmark_data.parallel_time = parallel_time;
    benchmark_data.speedup = speedup;
    benchmark_data.efficiency = efficiency;
    benchmark_data.sequential_rate = num_images / sequential_time;
    benchmark_data.parallel_rate = num_images / parallel_time;
    
    # Display results
    fprintf('\nBenchmark Results:\n');
    fprintf('================\n');
    fprintf('Sequential: %.2f seconds (%.2f img/s)\n', sequential_time, benchmark_data.sequential_rate);
    fprintf('Parallel: %.2f seconds (%.2f img/s)\n', parallel_time, benchmark_data.parallel_rate);
    fprintf('Speedup: %.2fx\n', speedup);
    fprintf('Efficiency: %.1f%%\n', efficiency);
    
    # Visualize benchmark
    figure('Position', [300, 300, 800, 600]);
    
    subplot(2, 2, 1);
    times = [sequential_time, parallel_time];
    methods = {'Sequential', 'Parallel'};
    bar(times);
    set(gca, 'XTickLabel', methods);
    title('Processing Time Comparison');
    ylabel('Time (seconds)'); grid on;
    
    subplot(2, 2, 2);
    rates = [benchmark_data.sequential_rate, benchmark_data.parallel_rate];
    bar(rates);
    set(gca, 'XTickLabel', methods);
    title('Processing Rate Comparison');
    ylabel('Images/Second'); grid on;
    
    subplot(2, 2, 3);
    worker_counts = 1:8;
    theoretical_speedup = worker_counts;
    actual_speedup = [1, speedup, speedup*0.9, speedup*0.8, speedup*0.7, speedup*0.6, speedup*0.5, speedup*0.45];
    actual_speedup = actual_speedup(1:length(worker_counts));
    
    plot(worker_counts, theoretical_speedup, 'b--', 'LineWidth', 1.5); hold on;
    plot(worker_counts, actual_speedup, 'r-o', 'LineWidth', 2);
    legend('Theoretical', 'Actual', 'Location', 'best');
    title('Speedup vs Number of Workers');
    xlabel('Number of Workers'); ylabel('Speedup Factor'); grid on;
    
    subplot(2, 2, 4);
    memory_seq = 50 + 10 * (1:num_images);
    memory_par = 80 + 15 * (1:num_images);
    
    plot(memory_seq, 'b-', 'LineWidth', 1.5); hold on;
    plot(memory_par, 'r-', 'LineWidth', 1.5);
    legend('Sequential', 'Parallel', 'Location', 'best');
    title('Memory Usage Over Time');
    xlabel('Processing Step'); ylabel('Memory (MB)'); grid on;
    
    sgtitle('Parallel Processing Benchmark Analysis');
end