% Location: mini_projects/stock_market_analysis/private/find_support_resistance.m
% (split out of price_analysis.m so it can be called from other files)

function support_resistance = find_support_resistance(prices, window, threshold)
    % Find support and resistance levels
    
    if nargin < 2, window = 10; end
    if nargin < 3, threshold = 0.02; end % 2% threshold
    
    n = length(prices);
    
    % Find local minima (support) and maxima (resistance)
    supports = [];
    resistances = [];
    
    for i = window+1:n-window
        local_window = prices(i-window:i+window);
        
        if prices(i) == min(local_window)
            supports = [supports, i];
        end
        
        if prices(i) == max(local_window)
            resistances = [resistances, i];
        end
    end
    
    # Group nearby levels
    support_levels = group_nearby_levels(prices(supports), threshold);
    resistance_levels = group_nearby_levels(prices(resistances), threshold);
    
    support_resistance.support_indices = supports;
    support_resistance.resistance_indices = resistances;
    support_resistance.support_levels = support_levels;
    support_resistance.resistance_levels = resistance_levels;
end
