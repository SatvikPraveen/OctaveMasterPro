% Location: mini_projects/parallel_image_batch_processing/private/get_operation_names.m
% (split out of performance_benchmark.m so it can be called from other files)

function operation_names = get_operation_names(operations)
    % Get operation names for display
    
    operation_names = cell(size(operations));
    
    for i = 1:length(operations)
        if ischar(operations{i}) || isstring(operations{i})
            operation_names{i} = operations{i};
        else
            operation_names{i} = sprintf('Custom_%d', i);
        end
    end
end
