% Location: mini_projects/image_processing_basics/private/adaptive_histogram_equalization.m
% (split out of histogram_analysis.m so it can be called from other files)

function enhanced_img = adaptive_histogram_equalization(img, varargin)
    % Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    % Default parameters
    tile_size = [8, 8];
    clip_limit = 0.02;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'tile_size'
                tile_size = varargin{i+1};
            case 'clip_limit'
                clip_limit = varargin{i+1};
        end
    end
    
    if size(img, 3) > 1, img = rgb2gray_custom(img); end
    
    if max(img(:)) <= 1
        img = img * 255;
        normalize_output = true;
    else
        normalize_output = false;
    end
    
    % Integer grey levels 0..255 are used as indices into the mapping
    img = round(min(max(double(img), 0), 255));
    
    [h, w] = size(img);
    tile_h = round(h / tile_size(1));
    tile_w = round(w / tile_size(2));
    
    enhanced_img = zeros(size(img));
    
    % Process each tile
    for tile_i = 1:tile_size(1)
        for tile_j = 1:tile_size(2)
            % Define tile boundaries
            row_start = (tile_i-1) * tile_h + 1;
            row_end = min(tile_i * tile_h, h);
            if tile_i == tile_size(1), row_end = h; end  % cover remainder rows
            col_start = (tile_j-1) * tile_w + 1;
            col_end = min(tile_j * tile_w, w);
            if tile_j == tile_size(2), col_end = w; end  % cover remainder columns
            
            % Extract tile
            tile = img(row_start:row_end, col_start:col_end);
            
            % Compute clipped histogram
            hist_counts = hist(double(tile(:)), 0:255);
            
            % Apply clipping
            clip_value = clip_limit * numel(tile) / 256;
            clipped_hist = min(hist_counts, clip_value);
            
            % Redistribute clipped pixels
            excess = sum(hist_counts - clipped_hist);
            redistribution = excess / 256;
            clipped_hist = clipped_hist + redistribution;
            
            % Compute CDF and mapping
            cdf = cumsum(clipped_hist);
            cdf = cdf / cdf(end);
            mapping = round(255 * cdf);
            
            % Apply equalization to tile
            for i_tile = 1:size(tile, 1)
                for j_tile = 1:size(tile, 2)
                    pixel_val = tile(i_tile, j_tile);
                    enhanced_img(row_start + i_tile - 1, col_start + j_tile - 1) = mapping(pixel_val + 1);
                end
            end
        end
    end
    
    if normalize_output
        enhanced_img = enhanced_img / 255;
    end
end
