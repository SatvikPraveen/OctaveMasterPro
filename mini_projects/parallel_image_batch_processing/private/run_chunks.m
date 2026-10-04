% Location: mini_projects/parallel_image_batch_processing/private/run_chunks.m

function [chunk_results, mode] = run_chunks(chunks, output_dir, operation_func, save_results, num_workers)
    % Process image chunks with real worker processes when the Octave
    % 'parallel' package is available (parcellfun), otherwise sequentially.
    % MODE reports what actually happened ('parallel (N workers)' or
    % 'sequential (<reason>)') so timings are never
    % presented as parallel when they were not.
    have_parallel = false;
    try
        pkg load parallel
        have_parallel = exist('parcellfun', 'file') > 0 && num_workers > 1;
    catch
        have_parallel = false;
    end
    reason = 'parallel package unavailable';
    if have_parallel && strncmp(func2str(operation_func), '@', 1)
        % Anonymous operations that call sub/private functions cannot be
        % resolved inside parcellfun workers; run them here instead.
        have_parallel = false;
        reason = 'anonymous operation cannot be dispatched to workers';
    end

    n = numel(chunks);
    if have_parallel
        % Pass a direct handle plus per-call argument cells: anonymous
        % functions that call private/sub-functions cannot be resolved in
        % parcellfun worker processes, whereas direct handles can.
        chunk_results = parcellfun(num_workers, @process_image_chunk, chunks, ...
                                   repmat({output_dir}, size(chunks)), ...
                                   repmat({operation_func}, size(chunks)), ...
                                   repmat({save_results}, size(chunks)), ...
                                   'UniformOutput', false, 'VerboseLevel', 0, ...
                                   'ErrorHandler', @worker_failed);
        mode = sprintf('parallel (%d workers)', num_workers);
    else
        chunk_results = cell(n, 1);
        for i = 1:n
            chunk_results{i} = process_image_chunk(chunks{i}, output_dir, operation_func, save_results);
        end
        mode = sprintf('sequential (%s)', reason);
    end
    chunk_results = chunk_results(:);
end

function r = worker_failed(err, chunk, varargin)
    % A failed worker counts every image of its chunk as an error instead
    % of aborting the whole batch; the message is reported, not hidden.
    fprintf(2, 'worker error (%s): %s\n', err.identifier, err.message);
    r.num_processed = 0;
    r.error_count = numel(chunk);
end
