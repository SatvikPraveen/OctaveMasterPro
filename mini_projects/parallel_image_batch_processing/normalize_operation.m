% Location: mini_projects/parallel_image_batch_processing/normalize_operation.m
% (split out of image_operations.m so it can be called from other files)

function normalized_img = normalize_operation(img, varargin)
    % Normalize image values
    
    target_range = [0, 1];
    method = 'minmax';
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'range', target_range = varargin{i+1};
            case 'method', method = varargin{i+1};
        end
    end
    
    switch lower(method)
        case 'minmax'
            min_val = min(img(:));
            max_val = max(img(:));
            if max_val > min_val
                normalized_img = (img - min_val) / (max_val - min_val);
                normalized_img = normalized_img * (target_range(2) - target_range(1)) + target_range(1);
            else
                normalized_img = img;
            end
            
        case 'zscore'
            mean_val = mean(img(:));
            std_val = std(img(:));
            if std_val > 0
                normalized_img = (img - mean_val) / std_val;
            else
                normalized_img = img;
            end
            
        case 'robust'
            # Use percentiles for robust normalization
            p5 = prctile(img(:), 5);
            p95 = prctile(img(:), 95);
            if p95 > p5
                normalized_img = (img - p5) / (p95 - p5);
                normalized_img = max(0, min(1, normalized_img));
            else
                normalized_img = img;
            end
    end
end
