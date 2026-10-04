% Location: mini_projects/image_processing_basics/private/non_maximum_suppression.m
% (split out of basic_filters.m so it can be called from other files)

function suppressed = non_maximum_suppression(magnitude, direction)
    % Non-maximum suppression for edge thinning
    
    [h, w] = size(magnitude);
    suppressed = zeros(h, w);
    
    % Convert direction to degrees and normalize to [0, 180)
    angle = mod(direction * 180 / pi + 180, 180);
    
    for i = 2:h-1
        for j = 2:w-1
            % Determine gradient direction
            if (angle(i,j) >= 0 && angle(i,j) < 22.5) || (angle(i,j) >= 157.5 && angle(i,j) <= 180)
                % Horizontal edge
                neighbors = [magnitude(i, j-1), magnitude(i, j+1)];
            elseif angle(i,j) >= 22.5 && angle(i,j) < 67.5
                % Diagonal edge (\)
                neighbors = [magnitude(i-1, j+1), magnitude(i+1, j-1)];
            elseif angle(i,j) >= 67.5 && angle(i,j) < 112.5
                % Vertical edge
                neighbors = [magnitude(i-1, j), magnitude(i+1, j)];
            else
                % Diagonal edge (/)
                neighbors = [magnitude(i-1, j-1), magnitude(i+1, j+1)];
            end
            
            % Suppress if not maximum
            if magnitude(i,j) >= max(neighbors)
                suppressed(i,j) = magnitude(i,j);
            end
        end
    end
end
