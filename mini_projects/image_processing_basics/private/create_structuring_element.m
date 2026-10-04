% Location: mini_projects/image_processing_basics/private/create_structuring_element.m
% (split out of morphology_ops.m so it can be called from other files)

function se = create_structuring_element(shape, size_param, varargin)
    % Create structuring elements for morphological operations
    
    switch lower(shape)
        case 'disk'
            radius = size_param;
            [x, y] = meshgrid(-radius:radius, -radius:radius);
            se = double((x.^2 + y.^2) <= radius^2);
            
        case 'square'
            side = size_param;
            se = ones(side, side);
            
        case 'cross'
            size_val = size_param;
            se = zeros(2*size_val+1, 2*size_val+1);
            se(size_val+1, :) = 1;
            se(:, size_val+1) = 1;
            
        case 'line'
            length_val = size_param;
            angle = 0;
            if length(varargin) >= 1, angle = varargin{1}; end
            
            if angle == 0
                se = ones(1, length_val);
            elseif angle == 90
                se = ones(length_val, 1);
            else
                angle_rad = angle * pi / 180;
                x_end = round(length_val * cos(angle_rad));
                y_end = round(length_val * sin(angle_rad));
                max_dim = max(abs(x_end), abs(y_end)) + 1;
                se = zeros(2*max_dim+1, 2*max_dim+1);
                center = max_dim + 1;
                for t = 0:0.1:1
                    x = round(center + t * x_end);
                    y = round(center + t * y_end);
                    if x >= 1 && x <= size(se, 2) && y >= 1 && y <= size(se, 1)
                        se(y, x) = 1;
                    end
                end
            end
            
        case 'diamond'
            radius = size_param;
            [x, y] = meshgrid(-radius:radius, -radius:radius);
            se = double((abs(x) + abs(y)) <= radius);
    end
end
