% Location: mini_projects/stock_market_analysis/private/exponential_moving_average.m
% (split out of technical_indicators.m so it can be called from other files)

function ema = exponential_moving_average(prices, window)
    % Exponential Moving Average
    
    n = length(prices);
    ema = NaN(n, 1);
    alpha = 2 / (window + 1);
    
    % Initialize with SMA
    ema(window) = mean(prices(1:window));
    
    for i = window+1:n
        ema(i) = alpha * prices(i) + (1 - alpha) * ema(i-1);
    end
end
