% Location: mini_projects/image_processing_basics/private/load_image_batch.m
% (split out of image_loader.m so it can be called from other files)

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
