% Location: mini_projects/stock_market_analysis/private/williams_percent_r.m
% (split out of technical_indicators.m so it can be called from other files)

function williams_r = williams_percent_r(high, low, close, period)
    % Williams %R Oscillator
    
    n = length(close);
    williams_r = NaN(n, 1);
    
    for i = period:n
        highest_high = max(high(i-period+1:i));
        lowest_low = min(low(i-period+1:i));
        
        if highest_high > lowest_low
            williams_r(i) = -100 * (highest_high - close(i)) / (highest_high - lowest_low);
        else
            williams_r(i) = -50;
        end
    end
end
