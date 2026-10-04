% Location: mini_projects/parallel_image_batch_processing/private/prctile.m
% (split out of image_operations.m so it can be called from other files)

function p = prctile(data, percentile)
    % Calculate percentile
    sorted_data = sort(data(:));
    n = length(sorted_data);
    index = percentile/100 * (n - 1) + 1;
    
    if index == round(index)
        p = sorted_data(round(index));
    else
        lower = floor(index);
        upper = ceil(index);
        weight = index - lower;
        p = sorted_data(lower) * (1 - weight) + sorted_data(upper) * weight;
    end
end
