% Location: mini_projects/stock_market_analysis/private/calculate_kurtosis.m
% (split out of price_analysis.m so it can be called from other files)

function kurt = calculate_kurtosis(data)
    mu = mean(data);
    sigma = std(data);
    kurt = mean(((data - mu) / sigma).^4) - 3;
end
