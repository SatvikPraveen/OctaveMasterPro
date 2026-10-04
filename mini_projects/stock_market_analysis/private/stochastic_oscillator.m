% Location: mini_projects/stock_market_analysis/private/stochastic_oscillator.m
% (split out of technical_indicators.m so it can be called from other files)

function stoch = stochastic_oscillator(high, low, close, k_period, d_period)
    % Stochastic Oscillator
    
    n = length(close);
    percent_k = NaN(n, 1);
    
    for i = k_period:n
        highest_high = max(high(i-k_period+1:i));
        lowest_low = min(low(i-k_period+1:i));
        
        if highest_high > lowest_low
            percent_k(i) = 100 * (close(i) - lowest_low) / (highest_high - lowest_low);
        else
            percent_k(i) = 50; % Neutral value
        end
    end
    
    % %D is SMA of %K
    percent_d = simple_moving_average(percent_k, d_period);
    
    stoch.percent_k = percent_k;
    stoch.percent_d = percent_d;
end
