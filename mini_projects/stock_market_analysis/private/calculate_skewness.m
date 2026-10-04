% Location: mini_projects/stock_market_analysis/private/calculate_skewness.m
% (split out of price_analysis.m so it can be called from other files)

function skew = calculate_skewness(data)
    mu = mean(data);
    sigma = std(data);
    skew = mean(((data - mu) / sigma).^3);
end
