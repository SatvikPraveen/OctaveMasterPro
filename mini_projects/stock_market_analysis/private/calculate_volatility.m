% Location: mini_projects/stock_market_analysis/private/calculate_volatility.m
% (split out of price_analysis.m so it can be called from other files)

function volatility = calculate_volatility(returns, window, method)
    % Calculate various volatility measures
    
    if nargin < 2, window = 20; end
    if nargin < 3, method = 'standard'; end
    
    n = length(returns);
    volatility = NaN(n, 1);
    
    switch lower(method)
        case 'standard'
            % Standard deviation of returns
            for i = window:n
                volatility(i) = std(returns(i-window+1:i));
            end
            
        case 'exponential'
            % Exponentially weighted volatility
            lambda = 0.94; % Decay factor
            volatility(window) = std(returns(1:window));
            
            for i = window+1:n
                volatility(i) = sqrt(lambda * volatility(i-1)^2 + (1-lambda) * returns(i)^2);
            end
            
        case 'garman_klass'
            % Garman-Klass volatility (requires OHLC data)
            fprintf('Garman-Klass requires OHLC data in main function call.\n');
            volatility = calculate_volatility(returns, window, 'standard');
    end
    
    % Annualize volatility
    volatility = volatility * sqrt(252);
end
