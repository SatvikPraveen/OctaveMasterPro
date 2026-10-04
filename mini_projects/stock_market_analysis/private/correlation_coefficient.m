% Location: mini_projects/stock_market_analysis/private/correlation_coefficient.m
% (split out of price_analysis.m so it can be called from other files)

function corr_coef = correlation_coefficient(x, y)
    % Calculate correlation coefficient
    
    if length(x) ~= length(y)
        error('Input vectors must have same length');
    end
    
    # Remove NaN values
    valid_idx = ~isnan(x) & ~isnan(y);
    x = x(valid_idx);
    y = y(valid_idx);
    
    if length(x) < 2
        corr_coef = NaN;
        return;
    end
    
    x_centered = x - mean(x);
    y_centered = y - mean(y);
    
    numerator = sum(x_centered .* y_centered);
    denominator = sqrt(sum(x_centered.^2) * sum(y_centered.^2));
    
    if denominator == 0
        corr_coef = NaN;
    else
        corr_coef = numerator / denominator;
    end
end
