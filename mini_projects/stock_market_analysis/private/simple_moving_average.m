% Location: mini_projects/stock_market_analysis/private/simple_moving_average.m
% (split out of technical_indicators.m so it can be called from other files)

function sma = simple_moving_average(prices, window)
    % Simple Moving Average
    
    n = length(prices);
    sma = NaN(n, 1);
    
    for i = window:n
        sma(i) = mean(prices(i-window+1:i));
    end
end
