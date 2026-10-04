% Location: mini_projects/parallel_image_batch_processing/create_thumbnail.m
% (split out of image_operations.m so it can be called from other files)

function thumbnail_img = create_thumbnail(img, varargin)
    % Create thumbnail with optional border and metadata
    
    thumb_size = [128, 128];
    add_border = true;
    border_color = [0.5, 0.5, 0.5];
    border_width = 2;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'size', thumb_size = varargin{i+1};
            case 'border', add_border = varargin{i+1};
            case 'border_color', border_color = varargin{i+1};
            case 'border_width', border_width = varargin{i+1};
        end
    end
    
    # Resize to thumbnail size
    thumbnail_img = resize_operation(img, 'size', thumb_size);
    
    # Add border if requested
    if add_border
        [h, w, c] = size(thumbnail_img);
        
        if c == 1
            # Grayscale border
            thumbnail_img(1:border_width, :) = border_color(1);
            thumbnail_img(end-border_width+1:end, :) = border_color(1);
            thumbnail_img(:, 1:border_width) = border_color(1);
            thumbnail_img(:, end-border_width+1:end) = border_color(1);
        else
            # Color border
            for ch = 1:c
                thumbnail_img(1:border_width, :, ch) = border_color(min(ch, length(border_color)));
                thumbnail_img(end-border_width+1:end, :, ch) = border_color(min(ch, length(border_color)));
                thumbnail_img(:, 1:border_width, ch) = border_color(min(ch, length(border_color)));
                thumbnail_img(:, end-border_width+1:end, ch) = border_color(min(ch, length(border_color)));
            end
        end
    end
end
