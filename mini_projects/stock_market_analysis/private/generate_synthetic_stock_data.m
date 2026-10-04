% Location: mini_projects/stock_market_analysis/private/generate_synthetic_stock_data.m
% (split out of data_loader.m so it can be called from other files)

function synthetic_data = generate_synthetic_stock_data(varargin)
    % Generate realistic synthetic stock data
    
    num_days = 252;
    initial_price = 100;
    annual_return = 0.08;
    annual_volatility = 0.2;
    seed = 123;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'days', num_days = varargin{i+1};
            case 'initial_price', initial_price = varargin{i+1};
            case 'return', annual_return = varargin{i+1};
            case 'volatility', annual_volatility = varargin{i+1};
            case 'seed', seed = varargin{i+1};
        end
    end
    
    daily_return = annual_return / 252;
    daily_volatility = annual_volatility / sqrt(252);
    
    % Seed per call so results are reproducible.  Callers generating several
    % stocks must pass distinct seeds; a fixed seed made every stock share
    % one return path (correlation 1), invalidating portfolio analysis.
    rand('state', seed); randn('state', seed);
    returns = daily_return + daily_volatility * randn(num_days, 1);
    
    log_prices = log(initial_price) + cumsum(returns);
    close_prices = exp(log_prices);
    
    synthetic_data.close = close_prices;
    
    high_factor = 1 + abs(0.02 * randn(num_days, 1));
    synthetic_data.high = close_prices .* high_factor;
    
    low_factor = 1 - abs(0.015 * randn(num_days, 1));
    synthetic_data.low = close_prices .* low_factor;
    
    synthetic_data.open = [initial_price; close_prices(1:end-1)] .* (1 + 0.005 * randn(num_days, 1));
    
    for i = 1:num_days
        high_val = max([synthetic_data.open(i), synthetic_data.close(i)]);
        low_val = min([synthetic_data.open(i), synthetic_data.close(i)]);
        
        synthetic_data.high(i) = max(synthetic_data.high(i), high_val);
        synthetic_data.low(i) = min(synthetic_data.low(i), low_val);
    end
    
    price_changes = abs(diff([initial_price; close_prices]));
    base_volume = 1000000;
    volume_multiplier = 1 + 2 * (price_changes / mean(price_changes));
    synthetic_data.volume = base_volume * volume_multiplier .* (1 + 0.3 * randn(num_days, 1));
    synthetic_data.volume = max(synthetic_data.volume, base_volume * 0.1);
    
    synthetic_data.adj_close = synthetic_data.close;
    synthetic_data.dates = generate_date_vector(num_days);
    
    fprintf('Generated synthetic stock data: %d days\n', num_days);
end
