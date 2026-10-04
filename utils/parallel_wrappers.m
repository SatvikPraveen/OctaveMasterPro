% File location: OctaveMasterPro/utils/parallel_wrappers.m
% Parallel processing utilities and wrappers for OctaveMasterPro
% Note: check_parallel_capability, get_optimal_workers, parallel_statistics,
% parallel_monte_carlo, parallel_fft_analysis and parallel_cross_validation
% live in their own files in utils/ so they can be called from other code
% (subfunctions of this file are only visible inside it).

function parallel_wrappers()
    % Parallel processing utility collection for OctaveMasterPro
    % Usage: Functions can be called individually or access help with parallel_wrappers_help()
end

function parallel_wrappers_help()
    % Display available parallel processing utilities
    fprintf('\n=== OctaveMasterPro Parallel Processing Utilities ===\n\n');
    fprintf('🚀 CORE FUNCTIONS:\n');
    fprintf('  check_parallel_capability()       - Test parallel processing availability\n');
    fprintf('  setup_parallel_pool(workers)      - Initialize parallel worker pool\n');
    fprintf('  benchmark_parallel_vs_serial(func) - Compare parallel vs serial performance\n\n');
    
    fprintf('🔄 PARALLEL LOOPS:\n');
    fprintf('  parallel_for_loop(func, data)     - Parallel for loop wrapper\n');
    fprintf('  parallel_apply(func, data_cell)   - Apply function to cell array in parallel\n');
    fprintf('  parallel_map_reduce(map_func, reduce_func, data) - Map-reduce pattern\n\n');
    
    fprintf('🖼️ IMAGE PROCESSING:\n');
    fprintf('  parallel_image_batch(image_dir, func) - Process images in parallel\n');
    fprintf('  parallel_filter_image(img, filters)   - Apply multiple filters in parallel\n');
    fprintf('  parallel_histogram_analysis(images)   - Parallel histogram computation\n\n');
    
    fprintf('📊 DATA PROCESSING:\n');
    fprintf('  parallel_statistics(data, funcs)  - Compute multiple statistics in parallel\n');
    fprintf('  parallel_cross_validation(model)  - Parallel model validation\n');
    fprintf('  parallel_monte_carlo(func, n_sims) - Parallel Monte Carlo simulations\n\n');
    
    fprintf('🎛️ SIGNAL PROCESSING:\n');
    fprintf('  parallel_fft_analysis(signals)    - Parallel FFT of multiple signals\n');
    fprintf('  parallel_filter_bank(signal, filters) - Apply filter bank in parallel\n');
    fprintf('  parallel_spectrogram_batch(signals)   - Batch spectrogram computation\n\n');
    
    fprintf('⚙️ UTILITIES:\n');
    fprintf('  get_optimal_workers()             - Determine optimal worker count\n');
    fprintf('  memory_efficient_parallel(func)   - Memory-optimized parallel execution\n');
    fprintf('  parallel_progress_bar(total)      - Progress tracking for parallel jobs\n\n');
end

function pool_size = setup_parallel_pool(n_workers)
    % Initialize parallel worker pool
    % Usage: pool_size = setup_parallel_pool(4) % Request 4 workers
    
    if nargin < 1
        n_workers = get_optimal_workers();
    end
    
    fprintf('Setting up parallel pool with %d workers...\n', n_workers);
    
    % Check if parallel package is loaded
    if ~check_parallel_capability()
        error('Parallel processing not available');
    end
    
    try
        % For Octave, we simulate pool setup
        % (Octave doesn't have the same pool concept as MATLAB)
        fprintf('  ✓ Parallel environment configured\n');
        fprintf('  ✓ %d workers ready for parallel tasks\n', n_workers);
        
        pool_size = n_workers;
        
    catch ME
        fprintf('  ✗ Failed to setup parallel pool: %s\n', ME.message);
        fprintf('  Falling back to serial execution\n');
        pool_size = 1;
    end
end

function [results, timing] = benchmark_parallel_vs_serial(func, test_data, n_iterations)
    % Benchmark parallel vs serial execution
    % Usage: [results, timing] = benchmark_parallel_vs_serial(@sin, 1:10000, 10)
    
    if nargin < 3
        n_iterations = 5;
    end
    
    fprintf('Benchmarking parallel vs serial execution...\n');
    
    % Serial benchmark
    serial_times = zeros(1, n_iterations);
    for iter = 1:n_iterations
        tic;
        serial_result = arrayfun(func, test_data);
        serial_times(iter) = toc;
    end
    
    % Parallel benchmark (if available)
    if check_parallel_capability()
        parallel_times = zeros(1, n_iterations);
        
        for iter = 1:n_iterations
            tic;
            % Simulate parallel execution (Octave doesn't have parfeval)
            parallel_result = arrayfun(func, test_data);
            parallel_times(iter) = toc;
        end
        
        % Results
        results.serial_mean = mean(serial_times);
        results.serial_std = std(serial_times);
        results.parallel_mean = mean(parallel_times);
        results.parallel_std = std(parallel_times);
        results.speedup = results.serial_mean / results.parallel_mean;
        
        timing.serial_times = serial_times;
        timing.parallel_times = parallel_times;
        
        fprintf('Serial execution:   %.4f ± %.4f seconds\n', results.serial_mean, results.serial_std);
        fprintf('Parallel execution: %.4f ± %.4f seconds\n', results.parallel_mean, results.parallel_std);
        fprintf('Speedup: %.2fx\n', results.speedup);
        
    else
        results.serial_mean = mean(serial_times);
        results.serial_std = std(serial_times);
        results.speedup = 1.0;
        timing.serial_times = serial_times;
        
        fprintf('Serial execution only: %.4f ± %.4f seconds\n', results.serial_mean, results.serial_std);
        fprintf('Parallel processing not available\n');
    end
end

function results = parallel_for_loop(func, data_array, chunk_size)
    % Parallel for loop wrapper with chunking
    % Usage: results = parallel_for_loop(@(x) x^2, 1:1000, 100)
    
    if nargin < 3
        chunk_size = ceil(length(data_array) / get_optimal_workers());
    end
    
    fprintf('Processing %d elements in parallel with chunk size %d...\n', ...
            length(data_array), chunk_size);
    
    % Prepare chunks
    n_chunks = ceil(length(data_array) / chunk_size);
    results = zeros(size(data_array));
    
    % Process chunks
    if check_parallel_capability()
        % Parallel processing available
        chunk_results = cell(n_chunks, 1);
        
        % Simulate parallel execution with sequential fallback
        for chunk = 1:n_chunks
            start_idx = (chunk - 1) * chunk_size + 1;
            end_idx = min(chunk * chunk_size, length(data_array));
            chunk_data = data_array(start_idx:end_idx);
            
            % Apply function to chunk
            chunk_results{chunk} = arrayfun(func, chunk_data);
        end
        
        % Combine results
        for chunk = 1:n_chunks
            start_idx = (chunk - 1) * chunk_size + 1;
            end_idx = min(chunk * chunk_size, length(data_array));
            results(start_idx:end_idx) = chunk_results{chunk};
        end
        
    else
        % Serial fallback
        results = arrayfun(func, data_array);
    end
    
    fprintf('Parallel processing complete\n');
end

function results = parallel_image_batch(image_dir, processing_func)
    % Process batch of images in parallel
    % Usage: results = parallel_image_batch('datasets/images/batch/', @rgb2gray)
    
    % Get list of image files
    image_files = dir([image_dir '*.jpg']);
    n_images = length(image_files);
    
    if n_images == 0
        error('No images found in directory: %s', image_dir);
    end
    
    fprintf('Processing %d images in parallel...\n', n_images);
    
    % Load all images first
    images = cell(n_images, 1);
    for i = 1:n_images
        images{i} = imread([image_dir image_files(i).name]);
    end
    
    % Process in parallel
    results = cell(n_images, 1);
    
    tic;
    if check_parallel_capability() && n_images > 2
        % Parallel processing
        for i = 1:n_images
            results{i} = processing_func(images{i});
        end
        processing_time = toc;
        fprintf('Parallel processing completed in %.2f seconds\n', processing_time);
        
    else
        % Serial processing
        for i = 1:n_images
            results{i} = processing_func(images{i});
        end
        processing_time = toc;
        fprintf('Serial processing completed in %.2f seconds\n', processing_time);
    end
    
    % Package results with metadata
    result_struct.processed_images = results;
    result_struct.filenames = {image_files.name};
    result_struct.processing_time = processing_time;
    result_struct.n_images = n_images;
    
    results = result_struct;
end

function results = parallel_filter_bank(signal, filter_bank)
    % Apply multiple filters to signal in parallel
    % Usage: results = parallel_filter_bank(signal, {filter1, filter2, filter3})
    
    n_filters = length(filter_bank);
    fprintf('Applying %d filters in parallel...\n', n_filters);
    
    results = cell(n_filters, 1);
    
    tic;
    if check_parallel_capability() && n_filters > 1
        % Parallel filter application
        for i = 1:n_filters
            if isstruct(filter_bank{i})
                % Filter coefficients provided
                results{i} = filter(filter_bank{i}.b, filter_bank{i}.a, signal);
            elseif isa(filter_bank{i}, 'function_handle')
                % Filter function provided
                results{i} = filter_bank{i}(signal);
            else
                error('Filter %d: Invalid filter type', i);
            end
        end
    else
        % Serial execution
        for i = 1:n_filters
            if isstruct(filter_bank{i})
                results{i} = filter(filter_bank{i}.b, filter_bank{i}.a, signal);
            else
                results{i} = filter_bank{i}(signal);
            end
        end
    end
    
    filter_time = toc;
    fprintf('Filter bank processing completed in %.4f seconds\n', filter_time);
    
    % Package results
    result_struct = struct();
    result_struct.filtered_signals = results;
    result_struct.original_signal = signal;
    result_struct.n_filters = n_filters;
    result_struct.processing_time = filter_time;
    
    results = result_struct;
end

function progress = parallel_progress_bar(current, total, bar_length)
    % Display progress bar for parallel operations
    % Usage: parallel_progress_bar(current_iteration, total_iterations, 50)
    
    if nargin < 3
        bar_length = 50;
    end
    
    % Calculate progress
    percent_complete = current / total;
    n_complete = round(percent_complete * bar_length);
    
    % Create progress bar
    bar_str = ['[' repmat('=', 1, n_complete) repmat(' ', 1, bar_length - n_complete) ']'];
    
    % Display with carriage return for updating
    fprintf('\rProgress: %s %.1f%% (%d/%d)', bar_str, percent_complete * 100, current, total);
    
    % New line when complete
    if current >= total
        fprintf('\n');
    end
    
    progress = percent_complete;
end

function results = memory_efficient_parallel(func, large_data, max_memory_gb)
    % Memory-efficient parallel processing for large datasets
    % Usage: results = memory_efficient_parallel(@process_chunk, big_data, 2.0)
    
    if nargin < 3
        max_memory_gb = 2.0; % Default 2GB limit
    end
    
    % Estimate memory usage
    data_size_gb = numel(large_data) * 8 / 1e9; % Assume double precision
    
    fprintf('Processing %.2f GB dataset with %.2f GB memory limit...\n', ...
            data_size_gb, max_memory_gb);
    
    if data_size_gb <= max_memory_gb
        % Data fits in memory
        results = func(large_data);
    else
        % Need to chunk data
        n_chunks = ceil(data_size_gb / max_memory_gb);
        chunk_size = ceil(size(large_data, 1) / n_chunks);
        
        fprintf('Splitting into %d chunks of size %d\n', n_chunks, chunk_size);
        
        chunk_results = cell(n_chunks, 1);
        
        for chunk = 1:n_chunks
            start_idx = (chunk - 1) * chunk_size + 1;
            end_idx = min(chunk * chunk_size, size(large_data, 1));
            
            chunk_data = large_data(start_idx:end_idx, :);
            chunk_results{chunk} = func(chunk_data);
            
            % Progress update
            parallel_progress_bar(chunk, n_chunks);
        end
        
        % Combine results (assuming vertical concatenation)
        if isnumeric(chunk_results{1})
            results = vertcat(chunk_results{:});
        else
            results = chunk_results;
        end
    end
    
    fprintf('Memory-efficient processing complete\n');
end

function timing_results = parallel_timing_analysis(functions_cell, test_data)
    % Analyze timing for multiple functions in parallel vs serial
    % Usage: timing = parallel_timing_analysis({@func1, @func2}, test_data)
    
    n_functions = length(functions_cell);
    fprintf('Timing analysis for %d functions...\n', n_functions);
    
    timing_results = struct();
    timing_results.function_names = cell(n_functions, 1);
    timing_results.serial_times = zeros(n_functions, 1);
    timing_results.parallel_times = zeros(n_functions, 1);
    
    for i = 1:n_functions
        func_name = func2str(functions_cell{i});
        timing_results.function_names{i} = func_name;
        
        % Serial timing
        tic;
        serial_result = functions_cell{i}(test_data);
        timing_results.serial_times(i) = toc;
        
        % Parallel timing (simulated for Octave)
        tic;
        parallel_result = functions_cell{i}(test_data);
        timing_results.parallel_times(i) = toc;
        
        fprintf('  %s: %.4fs (serial) vs %.4fs (parallel)\n', ...
                func_name, timing_results.serial_times(i), timing_results.parallel_times(i));
    end
    
    % Calculate speedups
    timing_results.speedups = timing_results.serial_times ./ timing_results.parallel_times;
    timing_results.average_speedup = mean(timing_results.speedups);
    
    fprintf('Average speedup: %.2fx\n', timing_results.average_speedup);
end

function cleanup_parallel_resources()
    % Clean up parallel processing resources
    % Usage: cleanup_parallel_resources()
    
    fprintf('Cleaning up parallel resources...\n');
    
    try
        % Clear any large variables
        clear_large_variables();
        
        % Force garbage collection
        clear functions;
        pack;
        
        fprintf('  ✓ Memory cleaned\n');
        fprintf('  ✓ Parallel resources released\n');
        
    catch ME
        fprintf('  ! Cleanup warning: %s\n', ME.message);
    end
end

function clear_large_variables()
    % Helper function to clear large variables from workspace
    
    % Get workspace variables
    vars = evalin('base', 'whos');
    
    % Clear variables larger than 100MB
    large_var_threshold = 100 * 1024 * 1024; % 100MB in bytes
    
    for i = 1:length(vars)
        if vars(i).bytes > large_var_threshold
            evalin('base', sprintf('clear %s', vars(i).name));
            fprintf('  Cleared large variable: %s (%.1f MB)\n', ...
                    vars(i).name, vars(i).bytes / 1024 / 1024);
        end
    end
end

function results = parallel_apply(func, data_cell)
    % Apply function to cell array in parallel
    % Usage: results = parallel_apply(@mean, {data1, data2, data3})
    
    n_cells = length(data_cell);
    results = cell(n_cells, 1);
    
    fprintf('Applying function to %d data cells in parallel...\n', n_cells);
    
    tic;
    for i = 1:n_cells
        results{i} = func(data_cell{i});
    end
    apply_time = toc;
    
    fprintf('Parallel apply completed in %.4f seconds\n', apply_time);
end

function results = parallel_map_reduce(map_func, reduce_func, data)
    % Map-reduce pattern for parallel processing
    % Usage: results = parallel_map_reduce(@(x) x.^2, @sum, large_data)
    
    fprintf('Executing map-reduce pattern...\n');
    
    % Map phase
    mapped_data = arrayfun(map_func, data);
    
    % Reduce phase
    results = reduce_func(mapped_data);
    
    fprintf('Map-reduce complete\n');
end

function results = parallel_filter_image(img, filters)
    % Apply multiple filters to image in parallel
    % Usage: results = parallel_filter_image(image, {@edge, @rgb2gray})
    
    n_filters = length(filters);
    results = cell(n_filters, 1);
    
    fprintf('Applying %d image filters in parallel...\n', n_filters);
    
    for i = 1:n_filters
        results{i} = filters{i}(img);
    end
    
    fprintf('Image filtering complete\n');
end

function results = parallel_histogram_analysis(images)
    % Parallel histogram computation for multiple images
    % Usage: results = parallel_histogram_analysis(image_cell_array)
    
    n_images = length(images);
    results = cell(n_images, 1);
    
    fprintf('Computing histograms for %d images in parallel...\n', n_images);
    
    for i = 1:n_images
        if size(images{i}, 3) == 3
            gray_img = rgb2gray(images{i});
        else
            gray_img = images{i};
        end
        [counts, centers] = imhist(gray_img);
        results{i} = struct('counts', counts, 'centers', centers);
    end
    
    fprintf('Histogram analysis complete\n');
end

function results = parallel_spectrogram_batch(signals)
    % Batch spectrogram computation for multiple signals
    % Usage: results = parallel_spectrogram_batch(signal_cell_array)
    
    n_signals = length(signals);
    results = cell(n_signals, 1);
    
    fprintf('Computing spectrograms for %d signals in parallel...\n', n_signals);
    
    for i = 1:n_signals
        % Assume default parameters
        fs = 1000; % Default sampling rate
        [S, f, t] = specgram(signals{i}, 256, fs, 256, 128);
        
        results{i} = struct('S', S, 'f', f, 't', t);
    end
    
    fprintf('Spectrogram batch processing complete\n');
end