% Location: mini_projects/parallel_image_batch_processing/parallel_demo.m
% Main Parallel Image Processing Demonstration

function parallel_demo()
    clear; clc; close all;
    
    fprintf('====================================================\n');
    fprintf('    PARALLEL IMAGE BATCH PROCESSING DEMO           \n');
    fprintf('====================================================\n\n');
    
    try
        while true
            fprintf('\nSelect a demonstration:\n');
            fprintf('1. Image Operations Demo\n');
            fprintf('2. Batch Processing Demo\n');
            fprintf('3. Performance Benchmarking\n');
            fprintf('4. Memory Usage Analysis\n');
            fprintf('5. Complete Parallel Pipeline\n');
            fprintf('6. Custom Operation Builder\n');
            fprintf('0. Exit\n');
            
            choice = input('Enter your choice (0-6): ');
            
            switch choice
                case 0
                    fprintf('\nExiting Parallel Processing Demo. Goodbye!\n');
                    break;
                case 1
                    image_operations_demo();
                case 2
                    batch_processing_demo();
                case 3
                    benchmarking_demo();
                case 4
                    memory_analysis_demo();
                case 5
                    complete_pipeline_demo();
                case 6
                    custom_operation_demo();
                otherwise
                    fprintf('Invalid choice. Please select 0-6.\n');
            end
            
            if choice ~= 0
                input('\nPress Enter to continue...');
            end
        end
        
    catch err
        fprintf('Error in parallel_demo: %s\n', err.message);
    end
end

function image_operations_demo()
    fprintf('\n--- Image Operations Demo ---\n');
    demo_image_operations();
end

function batch_processing_demo()
    fprintf('\n--- Batch Processing Demo ---\n');
    
    # Setup directories
    input_dir = 'temp_input_images';
    output_dir = 'temp_output_images';
    
    # Generate test images
    image_files = generate_test_images(input_dir, 16);
    
    fprintf('Testing batch processing with different operations...\n');
    
    # Test different operations
    operations = {@resize_operation, @filter_operation, @enhance_operation};
    operation_names = {'Resize', 'Filter', 'Enhance'};
    
    results_sequential = cell(length(operations), 1);
    results_parallel = cell(length(operations), 1);
    
    for op_idx = 1:length(operations)
        fprintf('\nTesting %s operation...\n', operation_names{op_idx});
        
        # Sequential processing
        fprintf('  Sequential processing...\n');
        results_sequential{op_idx} = process_image_batch(input_dir, [output_dir, '_seq'], ...
                                                       operations{op_idx}, 'parallel', false, 'progress', false);
        
        # Parallel processing
        fprintf('  Parallel processing...\n');
        results_parallel{op_idx} = process_image_batch(input_dir, [output_dir, '_par'], ...
                                                     operations{op_idx}, 'parallel', true, 'progress', false);
    end
    
    # Display comparison
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 2, 1);
    seq_times = cellfun(@(x) x.processing_time, results_sequential);
    par_times = cellfun(@(x) x.processing_time, results_parallel);
    
    x = 1:length(operations);
    bar(x-0.2, seq_times, 0.4); hold on;
    bar(x+0.2, par_times, 0.4);
    
    set(gca, 'XTickLabel', operation_names);
    legend('Sequential', 'Parallel', 'Location', 'best');
    title('Processing Time Comparison');
    ylabel('Time (seconds)'); grid on;
    
    subplot(2, 2, 2);
    speedups = seq_times ./ par_times;
    bar(speedups);
    set(gca, 'XTickLabel', operation_names);
    title('Parallel Speedup Factor');
    ylabel('Speedup (x)'); grid on;
    
    subplot(2, 2, 3);
    seq_rates = cellfun(@(x) x.images_per_second, results_sequential);
    par_rates = cellfun(@(x) x.images_per_second, results_parallel);
    
    bar(x-0.2, seq_rates, 0.4); hold on;
    bar(x+0.2, par_rates, 0.4);
    
    set(gca, 'XTickLabel', operation_names);
    legend('Sequential', 'Parallel', 'Location', 'best');
    title('Processing Rate Comparison');
    ylabel('Images/Second'); grid on;
    
    subplot(2, 2, 4);
    # Show sample processed images
    sample_img = load_image_safe(image_files{1});
    processed_sample = enhance_operation(sample_img);
    
    comparison = [sample_img, processed_sample];
    if size(comparison, 3) == 1
        imshow(comparison, []);
    else
        imshow(comparison);
    end
    title('Sample: Original | Enhanced');
    
    sgtitle('Batch Processing Performance Analysis');
    
    fprintf('\nBatch processing demonstration complete.\n');
    
    # Cleanup temporary directories
    cleanup_temp_dirs({input_dir, [output_dir, '_seq'], [output_dir, '_par']});
end

function benchmarking_demo()
    fprintf('\n--- Performance Benchmarking Demo ---\n');
    
    # Setup test environment
    input_dir = 'benchmark_images';
    test_sizes = [8, 16, 32, 64]; # Different batch sizes
    
    # Generate test images
    fprintf('Generating benchmark images...\n');
    max_images = max(test_sizes);
    image_files = generate_test_images(input_dir, max_images);
    
    # Benchmark different batch sizes
    seq_times = zeros(size(test_sizes));
    par_times = zeros(size(test_sizes));
    speedups = zeros(size(test_sizes));
    
    for i = 1:length(test_sizes)
        batch_size = test_sizes(i);
        fprintf('\nBenchmarking batch size: %d images\n', batch_size);
        
        test_files = image_files(1:batch_size);
        
        # Sequential benchmark
        tic;
        process_image_list(test_files, @enhance_operation, false);
        seq_times(i) = toc;
        
        # Parallel benchmark
        tic;
        process_image_list(test_files, @enhance_operation, true);
        par_times(i) = toc;
        
        speedups(i) = seq_times(i) / par_times(i);
        
        fprintf('  Sequential: %.2fs, Parallel: %.2fs, Speedup: %.2fx\n', ...
               seq_times(i), par_times(i), speedups(i));
    end
    
    # Display benchmark results
    figure('Position', [200, 200, 1200, 800]);
    
    subplot(2, 3, 1);
    plot(test_sizes, seq_times, 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
    plot(test_sizes, par_times, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
    legend('Sequential', 'Parallel', 'Location', 'best');
    title('Processing Time vs Batch Size');
    xlabel('Number of Images'); ylabel('Time (seconds)'); grid on;
    
    subplot(2, 3, 2);
    plot(test_sizes, speedups, 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
    title('Speedup Factor vs Batch Size');
    xlabel('Number of Images'); ylabel('Speedup Factor'); grid on;
    
    subplot(2, 3, 3);
    efficiency = speedups / 4 * 100; # Assuming 4 workers
    plot(test_sizes, efficiency, 'm-d', 'LineWidth', 2, 'MarkerSize', 8);
    title('Parallel Efficiency');
    xlabel('Number of Images'); ylabel('Efficiency (%)'); grid on;
    
    # Scalability analysis
    subplot(2, 3, 4);
    worker_counts = [1, 2, 4, 6, 8];
    theoretical_speedup = worker_counts;
    
    # Simulate realistic speedup curve
    amdahl_speedup = worker_counts ./ (0.2 + 0.8./worker_counts); # Amdahl's law with 20% serial
    
    plot(worker_counts, theoretical_speedup, 'b--', 'LineWidth', 1.5); hold on;
    plot(worker_counts, amdahl_speedup, 'r-', 'LineWidth', 2);
    
    legend('Ideal Speedup', 'Realistic Speedup', 'Location', 'best');
    title('Scalability Analysis');
    xlabel('Number of Workers'); ylabel('Speedup Factor'); grid on;
    
    # Memory usage simulation
    subplot(2, 3, 5);
    memory_seq = 50 + test_sizes * 2; # Linear growth
    memory_par = 80 + test_sizes * 1.5; # Slightly more overhead but better scaling
    
    plot(test_sizes, memory_seq, 'b-', 'LineWidth', 2); hold on;
    plot(test_sizes, memory_par, 'r-', 'LineWidth', 2);
    legend('Sequential', 'Parallel', 'Location', 'best');
    title('Memory Usage');
    xlabel('Batch Size'); ylabel('Memory (MB)'); grid on;
    
    # Throughput comparison
    subplot(2, 3, 6);
    throughput_seq = test_sizes ./ seq_times;
    throughput_par = test_sizes ./ par_times;
    
    plot(test_sizes, throughput_seq, 'b-', 'LineWidth', 2); hold on;
    plot(test_sizes, throughput_par, 'r-', 'LineWidth', 2);
    legend('Sequential', 'Parallel', 'Location', 'best');
    title('Throughput Analysis');
    xlabel('Batch Size'); ylabel('Images/Second'); grid on;
    
    sgtitle('Comprehensive Performance Benchmarking');
    
    fprintf('Performance benchmarking complete.\n');
    
    cleanup_temp_dirs({input_dir});
end

function process_image_list(image_files, operation_func, use_parallel)
    % Process list of images (helper for benchmarking)
    
    for i = 1:length(image_files)
        img = load_image_safe(image_files{i});
        if ~isempty(img)
            processed = operation_func(img);
            # Just process, don't save for benchmark
        end
    end
end

function memory_analysis_demo()
    fprintf('\n--- Memory Usage Analysis Demo ---\n');
    
    # Test memory usage with different image sizes and operations
    image_sizes = [100, 200, 400, 800];
    operations = {@resize_operation, @filter_operation, @enhance_operation};
    operation_names = {'Resize', 'Filter', 'Enhance'};
    
    memory_usage = zeros(length(image_sizes), length(operations));
    processing_times = zeros(length(image_sizes), length(operations));
    
    for size_idx = 1:length(image_sizes)
        img_size = image_sizes(size_idx);
        fprintf('Testing with %dx%d images...\n', img_size, img_size);
        
        # Create test image
        test_img = rand(img_size, img_size, 3);
        
        for op_idx = 1:length(operations)
            # Measure processing time
            tic;
            processed = operations{op_idx}(test_img);
            processing_times(size_idx, op_idx) = toc;
            
            # Simulate memory usage (proportional to image size)
            memory_usage(size_idx, op_idx) = img_size^2 * 3 * 8 / 1024^2; # MB
        end
    end
    
    # Display memory analysis
    figure('Position', [150, 150, 1200, 800]);
    
    subplot(2, 3, 1);
    plot(image_sizes.^2 / 1000, memory_usage);
    legend(operation_names, 'Location', 'best');
    title('Memory Usage vs Image Size');
    xlabel('Image Pixels (thousands)'); ylabel('Memory (MB)'); grid on;
    
    subplot(2, 3, 2);
    plot(image_sizes.^2 / 1000, processing_times);
    legend(operation_names, 'Location', 'best');
    title('Processing Time vs Image Size');
    xlabel('Image Pixels (thousands)'); ylabel('Time (seconds)'); grid on;
    
    # Memory efficiency
    subplot(2, 3, 3);
    efficiency = (image_sizes.^2)' ./ processing_times; # Pixels per second
    plot(image_sizes, efficiency);
    legend(operation_names, 'Location', 'best');
    title('Processing Efficiency');
    xlabel('Image Size'); ylabel('Pixels/Second'); grid on;
    
    # Memory vs performance tradeoff
    subplot(2, 3, 4);
    for op_idx = 1:length(operations)
        scatter(memory_usage(:, op_idx), 1./processing_times(:, op_idx), 100, 'filled');
        hold on;
    end
    legend(operation_names, 'Location', 'best');
    title('Memory vs Performance Tradeoff');
    xlabel('Memory Usage (MB)'); ylabel('Processing Rate (1/seconds)'); grid on;
    
    # Batch size optimization
    subplot(2, 3, 5);
    batch_sizes = [1, 2, 4, 8, 16, 32];
    optimal_batch_memory = zeros(size(batch_sizes));
    optimal_batch_time = zeros(size(batch_sizes));
    
    base_memory = 50; # Base memory overhead
    base_time = 0.1; # Base time overhead
    
    for i = 1:length(batch_sizes)
        # Simulate memory and time scaling
        optimal_batch_memory(i) = base_memory + batch_sizes(i) * 5; # Linear + overhead
        optimal_batch_time(i) = base_time + batch_sizes(i) * 0.05; # Sublinear scaling
    end
    
    [ax, h1, h2] = plotyy(batch_sizes, optimal_batch_memory, batch_sizes, optimal_batch_time);
    set(h1, 'LineStyle', '-o', 'LineWidth', 2, 'Color', 'blue');
    set(h2, 'LineStyle', '-s', 'LineWidth', 2, 'Color', 'red');
    ylabel(ax(1), 'Memory Usage (MB)');
    ylabel(ax(2), 'Processing Time (s)');
    xlabel('Batch Size');
    title('Batch Size Optimization');
    grid on;
    
    # Resource utilization
    subplot(2, 3, 6);
    workers = 1:8;
    cpu_util = min(100, workers * 15 - workers.^2 * 1.5); # Diminishing returns
    memory_util = 20 + workers * 8;
    
    [ax, h1, h2] = plotyy(workers, cpu_util, workers, memory_util);
    set(h1, 'LineStyle', '-o', 'LineWidth', 2, 'Color', 'green');
    set(h2, 'LineStyle', '-^', 'LineWidth', 2, 'Color', 'orange');
    ylabel(ax(1), 'CPU Utilization (%)');
    ylabel(ax(2), 'Memory Usage (MB)');
    xlabel('Number of Workers');
    title('Resource Utilization');
    grid on;
    
    sgtitle('Memory Usage and Performance Analysis');
    
    fprintf('Memory analysis demonstration complete.\n');
end

function benchmarking_demo()
    fprintf('\n--- Performance Benchmarking Demo ---\n');
    
    input_dir = 'benchmark_test_images';
    image_files = generate_test_images(input_dir, 24);
    
    # Test multiple operation functions
    benchmark_data = benchmark_processing_modes(image_files, @enhance_operation);
    
    cleanup_temp_dirs({input_dir});
end

function complete_pipeline_demo()
    fprintf('\n--- Complete Parallel Pipeline Demo ---\n');
    
    # Create comprehensive processing pipeline
    input_dir = 'pipeline_input';
    output_dir = 'pipeline_output';
    
    # Generate diverse test images
    fprintf('Setting up comprehensive test dataset...\n');
    image_files = generate_diverse_test_images(input_dir, 20);
    
    # Define multi-stage pipeline
    pipeline_operations = {
        @(img) resize_operation(img, 'size', [512, 512]);
        @(img) noise_reduction_operation(img, 'method', 'bilateral', 'strength', 0.8);
        @(img) enhance_operation(img, 'contrast', true, 'sharpness', true);
        @(img) add_watermark(img, 'text', 'PROCESSED', 'position', 'bottom-right');
    };
    
    operation_names = {'Resize to 512x512', 'Noise Reduction', 'Enhancement', 'Watermark'};
    
    fprintf('Executing %d-stage pipeline on %d images...\n', length(pipeline_operations), length(image_files));
    
    # Execute pipeline
    total_start_time = tic;
    stage_times = zeros(length(pipeline_operations), 1);
    
    current_images = cell(length(image_files), 1);
    
    # Load all images first
    fprintf('Loading images...\n');
    for i = 1:length(image_files)
        current_images{i} = load_image_safe(image_files{i});
    end
    
    # Process through pipeline stages
    for stage = 1:length(pipeline_operations)
        fprintf('Pipeline Stage %d: %s...\n', stage, operation_names{stage});
        
        stage_start = tic;
        
        # Process all images through current stage
        for img_idx = 1:length(current_images)
            if ~isempty(current_images{img_idx})
                current_images{img_idx} = pipeline_operations{stage}(current_images{img_idx});
            end
        end
        
        stage_times(stage) = toc(stage_start);
        
        fprintf('  Stage %d completed in %.2f seconds\n', stage, stage_times(stage));
    end
    
    total_pipeline_time = toc(total_start_time);
    
    # Save final results
    fprintf('Saving pipeline results...\n');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    for i = 1:length(current_images)
        if ~isempty(current_images{i})
            output_file = fullfile(output_dir, sprintf('pipeline_result_%03d.png', i));
            save_image_safe(current_images{i}, output_file);
        end
    end
    
    # Visualize pipeline performance
    figure('Position', [100, 100, 1400, 800]);
    
    subplot(2, 3, 1);
    bar(stage_times);
    set(gca, 'XTickLabel', 1:length(operation_names));
    title('Pipeline Stage Times');
    ylabel('Time (seconds)'); grid on;
    
    subplot(2, 3, 2);
    cumulative_times = cumsum(stage_times);
    plot(1:length(operation_names), cumulative_times, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
    title('Cumulative Processing Time');
    xlabel('Pipeline Stage'); ylabel('Cumulative Time (s)'); grid on;
    
    # Show sample progression through pipeline
    subplot(2, 3, [3, 6]);
    # Load sample image and process through stages
    sample_img = load_image_safe(image_files{1});
    pipeline_progression = cell(length(pipeline_operations) + 1, 1);
    pipeline_progression{1} = sample_img;
    
    current_sample = sample_img;
    for stage = 1:length(pipeline_operations)
        current_sample = pipeline_operations{stage}(current_sample);
        pipeline_progression{stage + 1} = current_sample;
    end
    
    # Create montage of pipeline progression
    montage_img = [];
    for i = 1:length(pipeline_progression)
        stage_img = pipeline_progression{i};
        
        # Resize for display
        if size(stage_img, 1) > 100 || size(stage_img, 2) > 100
            display_img = resize_operation(stage_img, 'size', [100, 100]);
        else
            display_img = stage_img;
        end
        
        if isempty(montage_img)
            montage_img = display_img;
        else
            montage_img = [montage_img, display_img];
        end
    end
    
    imshow(montage_img, []);
    title('Pipeline Progression: Original → Resize → Denoise → Enhance → Watermark');
    
    # Performance metrics
    subplot(2, 3, 4);
    metrics = {'Total Time', 'Avg Stage Time', 'Images/Sec'};
    values = [total_pipeline_time, mean(stage_times), length(image_files)/total_pipeline_time];
    
    bar(values);
    set(gca, 'XTickLabel', metrics);
    title('Pipeline Performance Metrics');
    ylabel('Values'); grid on;
    
    # Resource utilization over time
    subplot(2, 3, 5);
    time_points = cumsum([0; stage_times]);
    cpu_usage = [20, 60, 80, 70, 40]; # Simulated CPU usage per stage
    
    stairs(time_points, cpu_usage, 'LineWidth', 2);
    title('CPU Utilization Over Pipeline');
    xlabel('Time (seconds)'); ylabel('CPU Usage (%)'); grid on;
    
    sgtitle('Complete Parallel Processing Pipeline');
    
    fprintf('\nPipeline Results:\n');
    fprintf('================\n');
    fprintf('Total processing time: %.2f seconds\n', total_pipeline_time);
    fprintf('Average stage time: %.2f seconds\n', mean(stage_times));
    fprintf('Overall throughput: %.2f images/second\n', length(image_files)/total_pipeline_time);
    fprintf('Pipeline efficiency: %.1f%%\n', (length(image_files) * length(pipeline_operations)) / total_pipeline_time);
    
    cleanup_temp_dirs({input_dir, output_dir});
end

function custom_operation_demo()
    fprintf('\n--- Custom Operation Builder Demo ---\n');
    
    fprintf('This demo shows how to create custom parallel operations.\n');
    
    # Create sample custom operation
    custom_op = @(img) custom_artistic_filter(img);
    
    # Test the custom operation
    input_dir = 'custom_test_images';
    output_dir = 'custom_output_images';
    
    image_files = generate_test_images(input_dir, 8);
    
    fprintf('Testing custom artistic filter operation...\n');
    results = process_image_batch(input_dir, output_dir, custom_op, 'parallel', true);
    
    fprintf('Custom operation results:\n');
    fprintf('Processed %d images in %.2f seconds\n', results.num_images_processed, results.processing_time);
    
    cleanup_temp_dirs({input_dir, output_dir});
end

function artistic_img = custom_artistic_filter(img)
    % Custom artistic filter combining multiple effects
    
    # Apply multiple artistic effects
    
    # 1. Slight blur for smoothing
    blurred = apply_gaussian_blur(img, 1.2);
    
    # 2. Color enhancement
    enhanced = enhance_operation(blurred, 'contrast', true);
    
    # 3. Edge preservation
    if size(img, 3) > 1
        gray = 0.299*img(:,:,1) + 0.587*img(:,:,2) + 0.114*img(:,:,3);
    else
        gray = img;
    end
    
    edges = apply_edge_detection(gray);
    edge_mask = edges > 0.1;
    
    # 4. Combine effects
    artistic_img = enhanced;
    
    # Preserve edges by blending original detail back in edge areas
    if size(img, 3) == 1
        artistic_img(edge_mask) = 0.7 * enhanced(edge_mask) + 0.3 * img(edge_mask);
    else
        for c = 1:size(img, 3)
            artistic_img(edge_mask, c) = 0.7 * enhanced(edge_mask, c) + 0.3 * img(edge_mask, c);
        end
    end
end

function diverse_files = generate_diverse_test_images(output_dir, num_images)
    # Generate diverse test images for comprehensive testing
    
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    diverse_files = cell(num_images, 1);
    
    for i = 1:num_images
        # Create different types of test images
        img_type = mod(i, 5) + 1;
        img_size = 150 + round(150 * rand());
        
        switch img_type
            case 1 # Geometric patterns
                img = create_geometric_test_image(img_size);
            case 2 # Natural textures
                img = create_texture_test_image(img_size);
            case 3 # High contrast
                img = create_high_contrast_test_image(img_size);
            case 4 # Noisy images
                img = create_noisy_test_image(img_size);
            case 5 # Color gradients
                img = create_gradient_test_image(img_size);
        end
        
        filename = sprintf('diverse_test_%03d.png', i);
        filepath = fullfile(output_dir, filename);
        
        if save_image_safe(img, filepath)
            diverse_files{i} = filepath;
        end
    end
    
    diverse_files = diverse_files(~cellfun(@isempty, diverse_files));
end

function img = create_geometric_test_image(size_val)
    [x, y] = meshgrid(1:size_val, 1:size_val);
    img = zeros(size_val, size_val, 3);
    
    # Red: circles
    for i = 1:3
        cx = round(size_val * rand()); cy = round(size_val * rand());
        r = 10 + round(20 * rand());
        mask = (x-cx).^2 + (y-cy).^2 <= r^2;
        img(:,:,1) = img(:,:,1) + double(mask) * (0.3 + 0.7*rand());
    end
    
    # Green: rectangles
    for i = 1:2
        x1 = round(size_val * rand() * 0.7);
        y1 = round(size_val * rand() * 0.7);
        w = round(size_val * 0.2); h = round(size_val * 0.2);
        img(y1:y1+h, x1:x1+w, 2) = 0.5 + 0.5*rand();
    end
    
    # Blue: diagonal lines
    img(:,:,3) = 0.3 * sin(2*pi*(x+y)/50);
    
    img = max(0, min(1, img));
end

function img = create_texture_test_image(size_val)
    img = zeros(size_val, size_val, 3);
    
    [x, y] = meshgrid(1:size_val, 1:size_val);
    
    # Simulate natural texture with multiple frequency components
    for freq = [0.05, 0.1, 0.2]
        for c = 1:3
            phase = 2*pi*rand();
            img(:,:,c) = img(:,:,c) + 0.3 * sin(2*pi*freq*x + phase) .* cos(2*pi*freq*y + phase);
        end
    end
    
    # Add noise texture
    img = img + 0.2 * randn(size(img));
    img = max(0, min(1, img));
end

function img = create_high_contrast_test_image(size_val)
    img = zeros(size_val, size_val, 3);
    
    # Create high contrast patterns
    block_size = round(size_val / 8);
    
    for i = 1:8
        for j = 1:8
            y_start = (i-1)*block_size + 1;
            y_end = min(i*block_size, size_val);
            x_start = (j-1)*block_size + 1;
            x_end = min(j*block_size, size_val);
            
            if mod(i+j, 2) == 0
                img(y_start:y_end, x_start:x_end, :) = 1;
            end
        end
    end
end

function img = create_noisy_test_image(size_val)
    # Base pattern
    [x, y] = meshgrid(1:size_val, 1:size_val);
    img = 0.5 + 0.3 * sin(2*pi*x/30) .* cos(2*pi*y/30);
    
    # Add significant noise
    img = img + 0.4 * randn(size_val, size_val);
    
    # Make RGB
    img = repmat(img, [1, 1, 3]);
    img = max(0, min(1, img));
end

function img = create_gradient_test_image(size_val)
    [x, y] = meshgrid(1:size_val, 1:size_val);
    
    img = zeros(size_val, size_val, 3);
    img(:,:,1) = x / size_val; # Red gradient
    img(:,:,2) = y / size_val; # Green gradient
    img(:,:,3) = (x + y) / (2 * size_val); # Blue gradient
end

function cleanup_temp_dirs(dir_list)
    # Clean up temporary directories
    
    for i = 1:length(dir_list)
        if exist(dir_list{i}, 'dir')
            try
                rmdir(dir_list{i}, 's');
                fprintf('Cleaned up directory: %s\n', dir_list{i});
            catch
                fprintf('Could not clean up directory: %s\n', dir_list{i});
            end
        end
    end
end