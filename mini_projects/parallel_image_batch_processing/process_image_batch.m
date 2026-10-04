% Location: mini_projects/parallel_image_batch_processing/process_image_batch.m
% (split out of batch_processor.m so it can be called from other files)

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
    if results.use_parallel
        fprintf('Mode: Parallel\n');
    else
        fprintf('Mode: Sequential\n');
    end
    if use_parallel
        fprintf('Workers: %d\n', results.num_workers);
    end
    fprintf('Errors: %d\n', results.error_count);
end
