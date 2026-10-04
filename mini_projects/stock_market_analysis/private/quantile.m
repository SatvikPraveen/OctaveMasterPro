% Location: mini_projects/stock_market_analysis/private/quantile.m
% (split out of price_analysis.m so it can be called from other files)

function q = quantile(data, p)
    % Calculate quantile
    sorted_data = sort(data);
    n = length(sorted_data);
    index = p * (n - 1) + 1;
    
    if index == round(index)
        q = sorted_data(round(index));
    else
        lower = floor(index);
        upper = ceil(index);
        weight = index - lower;
        q = sorted_data(lower) * (1 - weight) + sorted_data(upper) * weight;
    end
end
