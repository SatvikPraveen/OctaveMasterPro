% Location: mini_projects/parallel_image_batch_processing/private/convert_operations_to_functions.m
% (split out of performance_benchmark.m so it can be called from other files)

function operation_funcs = convert_operations_to_functions(operations)
    % Convert operation names to function handles
    
    operation_funcs = cell(size(operations));
    
    for i = 1:length(operations)
        if ischar(operations{i}) || isstring(operations{i})
            switch lower(operations{i})
                case 'resize'
                    operation_funcs{i} = @(img) resize_operation(img, 'size', [256, 256]);
                case 'filter'
                    operation_funcs{i} = @(img) filter_operation(img, 'type', 'gaussian');
                case 'enhance'
                    operation_funcs{i} = @(img) enhance_operation(img);
                case 'edge'
                    operation_funcs{i} = @(img) filter_operation(img, 'type', 'edge');
                case 'denoise'
                    operation_funcs{i} = @(img) noise_reduction_operation(img);
                otherwise
                    operation_funcs{i} = @(img) img; % Identity operation
            end
        else
            operation_funcs{i} = operations{i};
        end
    end
end
