% Location: mini_projects/parallel_image_batch_processing/create_composite_operation.m
% (split out of image_operations.m so it can be called from other files)

function composite_img = create_composite_operation(img_list, varargin)
    % Create composite image from multiple images
    
    composition_type = 'average';
    weights = [];
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'type', composition_type = varargin{i+1};
            case 'weights', weights = varargin{i+1};
        end
    end
    
    if isempty(weights)
        weights = ones(length(img_list), 1) / length(img_list);
    end
    
    # Ensure all images are same size
    reference_size = size(img_list{1});
    for i = 2:length(img_list)
        if any(size(img_list{i}) ~= reference_size)
            img_list{i} = resize_operation(img_list{i}, 'size', reference_size(1:2));
        end
    end
    
    switch lower(composition_type)
        case 'average'
            composite_img = zeros(size(img_list{1}));
            for i = 1:length(img_list)
                composite_img = composite_img + weights(i) * img_list{i};
            end
            
        case 'maximum'
            composite_img = img_list{1};
            for i = 2:length(img_list)
                composite_img = max(composite_img, img_list{i});
            end
            
        case 'minimum'
            composite_img = img_list{1};
            for i = 2:length(img_list)
                composite_img = min(composite_img, img_list{i});
            end
    end
end
