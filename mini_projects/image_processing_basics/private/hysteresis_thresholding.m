% Location: mini_projects/image_processing_basics/private/hysteresis_thresholding.m
% (split out of basic_filters.m so it can be called from other files)

function edges = hysteresis_thresholding(img, low_thresh, high_thresh)
    % Hysteresis thresholding for Canny edge detection
    
    [h, w] = size(img);
    edges = zeros(h, w);
    
    % Strong edges (above high threshold)
    strong_edges = img > high_thresh;
    
    % Weak edges (between thresholds)
    weak_edges = (img >= low_thresh) & (img <= high_thresh);
    
    % Start with strong edges
    edges = strong_edges;
    
    % Connect weak edges to strong edges
    changed = true;
    while changed
        changed = false;
        old_edges = edges;
        
        for i = 2:h-1
            for j = 2:w-1
                if weak_edges(i,j) && ~edges(i,j)
                    % Check 8-connected neighborhood
                    neighborhood = edges(i-1:i+1, j-1:j+1);
                    if any(neighborhood(:))
                        edges(i,j) = 1;
                        changed = true;
                    end
                end
            end
        end
    end
end
