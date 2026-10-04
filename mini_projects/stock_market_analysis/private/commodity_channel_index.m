% Location: mini_projects/stock_market_analysis/private/commodity_channel_index.m
% (split out of technical_indicators.m so it can be called from other files)

function cci = commodity_channel_index(high, low, close, period)
    % Commodity Channel Index
    
    typical_price = (high + low + close) / 3;
    sma_tp = simple_moving_average(typical_price, period);
    
    n = length(close);
    cci = NaN(n, 1);
    
    for i = period:n
        mean_deviation = mean(abs(typical_price(i-period+1:i) - sma_tp(i)));
        if mean_deviation > 0
            cci(i) = (typical_price(i) - sma_tp(i)) / (0.015 * mean_deviation);
        end
    end
end
