% Location: mini_projects/stock_market_analysis/private/garman_klass_volatility.m
% (split out of price_analysis.m so it can be called from other files)

function gk_vol = garman_klass_volatility(high, low, open, close, window)
    % Garman-Klass volatility estimator
    
    n = length(close);
    gk_vol = NaN(n, 1);
    
    for i = window:n
        gk_values = NaN(window, 1);
        
        for j = i-window+1:i
            if j > 1
                log_hl = log(high(j) / low(j));
                log_co = log(close(j) / open(j));
                gk_values(j-i+window) = 0.5 * log_hl^2 - (2*log(2)-1) * log_co^2;
            end
        end
        
        gk_vol(i) = sqrt(mean(gk_values(~isnan(gk_values))) * 252);
    end
end
