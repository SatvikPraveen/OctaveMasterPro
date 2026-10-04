% Location: mini_projects/stock_market_analysis/private/bollinger_bands.m
% (split out of technical_indicators.m so it can be called from other files)

function [upper_band, middle_band, lower_band] = bollinger_bands(prices, period, num_std)
    % Bollinger Bands
    
    if nargin < 3, num_std = 2; end
    
    n = length(prices);
    upper_band = NaN(n, 1);
    middle_band = NaN(n, 1);
    lower_band = NaN(n, 1);
    
    for i = period:n
        window = prices(i-period+1:i);
        sma_val = mean(window);
        std_val = std(window);
        
        middle_band(i) = sma_val;
        upper_band(i) = sma_val + num_std * std_val;
        lower_band(i) = sma_val - num_std * std_val;
    end
end
