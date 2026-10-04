% Location: mini_projects/parallel_image_batch_processing/add_watermark.m
% (split out of image_operations.m so it can be called from other files)

function watermarked_img = add_watermark(img, varargin)
    % Add watermark to image
    
    watermark_text = 'PROCESSED';
    position = 'bottom-right';
    opacity = 0.3;
    font_size = 0.05; # Relative to image height
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'text', watermark_text = varargin{i+1};
            case 'position', position = varargin{i+1};
            case 'opacity', opacity = varargin{i+1};
            case 'font_size', font_size = varargin{i+1};
        end
    end
    
    [h, w, c] = size(img);
    watermarked_img = img;
    
    # Simple text watermark (basic implementation)
    text_height = round(h * font_size);
    text_width = length(watermark_text) * round(text_height * 0.6);
    
    # Determine position
    switch lower(position)
        case 'bottom-right'
            start_row = h - text_height - 10;
            start_col = w - text_width - 10;
        case 'bottom-left'
            start_row = h - text_height - 10;
            start_col = 10;
        case 'top-right'
            start_row = 10;
            start_col = w - text_width - 10;
        case 'top-left'
            start_row = 10;
            start_col = 10;
        case 'center'
            start_row = round(h/2 - text_height/2);
            start_col = round(w/2 - text_width/2);
    end
    
    # Ensure bounds
    start_row = max(1, min(h - text_height, start_row));
    start_col = max(1, min(w - text_width, start_col));
    
    # Create simple rectangular watermark
    end_row = min(h, start_row + text_height);
    end_col = min(w, start_col + text_width);
    
    if c == 1
        # Grayscale
        watermarked_img(start_row:end_row, start_col:end_col) = ...
            (1 - opacity) * watermarked_img(start_row:end_row, start_col:end_col) + opacity * 0.8;
    else
        # Color - add semi-transparent overlay
        for ch = 1:c
            watermarked_img(start_row:end_row, start_col:end_col, ch) = ...
                (1 - opacity) * watermarked_img(start_row:end_row, start_col:end_col, ch) + opacity * 0.8;
        end
    end
end
