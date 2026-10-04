% Location: mini_projects/parallel_image_batch_processing/private/create_processing_chunks.m
% (split out of batch_processor.m so it can be called from other files)

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
