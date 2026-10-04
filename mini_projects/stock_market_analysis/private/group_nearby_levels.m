% Location: mini_projects/stock_market_analysis/private/group_nearby_levels.m
% (split out of price_analysis.m so it can be called from other files)

function grouped_levels = group_nearby_levels(levels, threshold)
    % Group price levels that are close together
    
    if isempty(levels)
        grouped_levels = [];
        return;
    end
    
    sorted_levels = sort(levels);
    grouped_levels = [];
    current_group = [sorted_levels(1)];
    
    for i = 2:length(sorted_levels)
        if abs(sorted_levels(i) - mean(current_group)) / mean(current_group) < threshold
            current_group = [current_group, sorted_levels(i)];
        else
            grouped_levels = [grouped_levels, mean(current_group)];
            current_group = [sorted_levels(i)];
        end
    end
    
    # Add final group
    grouped_levels = [grouped_levels, mean(current_group)];
end
