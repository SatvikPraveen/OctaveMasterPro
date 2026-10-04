% Location: mini_projects/stock_market_analysis/private/average_true_range.m
% (split out of technical_indicators.m so it can be called from other files)

function atr = average_true_range(high, low, close, period)
    % Average True Range
    
    n = length(close);
    
    % Calculate true range
    tr = NaN(n, 1);
    for i = 2:n
        tr1 = high(i) - low(i);
        tr2 = abs(high(i) - close(i-1));
        tr3 = abs(low(i) - close(i-1));
        tr(i) = max([tr1, tr2, tr3]);
    end
    
    % Average true range
    atr = simple_moving_average(tr, period);
end
