% Location: mini_projects/stock_market_analysis/private/trend_analysis.m
% (split out of price_analysis.m so it can be called from other files)

function [trend, strength] = trend_analysis(prices, window)
    % Analyze price trends using linear regression
    
    if nargin < 2, window = 20; end
    
    n = length(prices);
    trend = NaN(n, 1);
    strength = NaN(n, 1);
    
    for i = window:n
        y = prices(i-window+1:i);
        x = (1:window)';
        
        % Linear regression
        X = [ones(window, 1), x];
        beta = (X' * X) \ (X' * y);
        
        trend(i) = beta(2); % Slope
        
        # Calculate R-squared
        y_pred = X * beta;
        ss_res = sum((y - y_pred).^2);
        ss_tot = sum((y - mean(y)).^2);
        strength(i) = 1 - ss_res / ss_tot; % R-squared
    end
end
