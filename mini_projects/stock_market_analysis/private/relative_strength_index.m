% Location: mini_projects/stock_market_analysis/private/relative_strength_index.m
% (split out of technical_indicators.m so it can be called from other files)

function rsi = relative_strength_index(prices, period)
    % Relative Strength Index
    
    n = length(prices);
    rsi = NaN(n, 1);
    
    % Calculate price changes
    price_changes = [NaN; diff(prices)];
    
    % Separate gains and losses
    gains = max(price_changes, 0);
    losses = -min(price_changes, 0);
    
    for i = period+1:n
        avg_gain = mean(gains(i-period+1:i));
        avg_loss = mean(losses(i-period+1:i));
        
        if avg_loss == 0
            rsi(i) = 100;
        else
            rs = avg_gain / avg_loss;
            rsi(i) = 100 - (100 / (1 + rs));
        end
    end
end
