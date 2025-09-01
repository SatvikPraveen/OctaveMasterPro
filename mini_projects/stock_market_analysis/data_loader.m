% Location: mini_projects/stock_market_analysis/data_loader.m
% Stock Data Loading and Preprocessing

function data = load_stock_data(filename, varargin)
    % Load stock data from CSV file
    
    % Default parameters
    validate_data = true;
    fill_method = 'forward';
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'validate'
                validate_data = varargin{i+1};
            case 'fill_missing'
                fill_method = varargin{i+1};
        end
    end
    
    try
        if exist(filename, 'file')
            fprintf('Loading stock data from: %s\n', filename);
            
            fid = fopen(filename, 'r');
            header_line = fgetl(fid);
            fclose(fid);
            
            headers = strsplit(header_line, ',');
            headers = cellfun(@(x) strtrim(x), headers, 'UniformOutput', false);
            
            raw_data = csvread(filename, 1, 0);
            
            col_map = map_columns(headers);
            
            data.open = raw_data(:, col_map.open);
            data.high = raw_data(:, col_map.high);
            data.low = raw_data(:, col_map.low);
            data.close = raw_data(:, col_map.close);
            
            if col_map.volume > 0
                data.volume = raw_data(:, col_map.volume);
            else
                data.volume = ones(size(data.close));
            end
            
            if col_map.adj_close > 0
                data.adj_close = raw_data(:, col_map.adj_close);
            else
                data.adj_close = data.close;
            end
            
            data.dates = generate_date_vector(length(data.close));
            
        else
            fprintf('File not found. Generating synthetic stock data...\n');
            data = generate_synthetic_stock_data();
        end
        
        if validate_data
            data = validate_stock_data(data);
        end
        
        if ~strcmp(fill_method, 'none')
            data = fill_missing_values(data, fill_method);
        end
        
        data = add_derived_fields(data);
        
        fprintf('Stock data loaded: %d trading days.\n', length(data.close));
        
    catch err
        fprintf('Error loading: %s\n', err.message);
        data = generate_synthetic_stock_data();
    end
end

function col_map = map_columns(headers)
    % Map CSV columns to standard field names
    
    col_map.open = find_column(headers, {'open'});
    col_map.high = find_column(headers, {'high'});
    col_map.low = find_column(headers, {'low'});
    col_map.close = find_column(headers, {'close'});
    col_map.volume = find_column(headers, {'volume', 'vol'});
    col_map.adj_close = find_column(headers, {'adj', 'adjusted'});
    
    if col_map.open == 0, col_map.open = 2; end
    if col_map.high == 0, col_map.high = 3; end
    if col_map.low == 0, col_map.low = 4; end
    if col_map.close == 0, col_map.close = 5; end
    if col_map.volume == 0, col_map.volume = 6; end
end

function col_idx = find_column(headers, keywords)
    col_idx = 0;
    for i = 1:length(headers)
        header_lower = lower(headers{i});
        for j = 1:length(keywords)
            if contains(header_lower, keywords{j})
                col_idx = i;
                return;
            end
        end
    end
end

function dates = generate_date_vector(num_days)
    start_date = datenum('2020-01-01');
    dates = start_date + (0:num_days-1);
end

function synthetic_data = generate_synthetic_stock_data(varargin)
    % Generate realistic synthetic stock data
    
    num_days = 252;
    initial_price = 100;
    annual_return = 0.08;
    annual_volatility = 0.2;
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'days', num_days = varargin{i+1};
            case 'initial_price', initial_price = varargin{i+1};
            case 'return', annual_return = varargin{i+1};
            case 'volatility', annual_volatility = varargin{i+1};
        end
    end
    
    daily_return = annual_return / 252;
    daily_volatility = annual_volatility / sqrt(252);
    
    rand('state', 123); randn('state', 123);
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

function validated_data = validate_stock_data(data)
    validated_data = data;
    issues_found = 0;
    
    if any(data.open <= 0) || any(data.high <= 0) || any(data.low <= 0) || any(data.close <= 0)
        valid_idx = data.open > 0 & data.high > 0 & data.low > 0 & data.close > 0;
        validated_data = filter_data_by_index(data, valid_idx);
        issues_found = issues_found + sum(~valid_idx);
    end
    
    ohlc_valid = validated_data.high >= validated_data.low & ...
                 validated_data.high >= validated_data.open & ...
                 validated_data.high >= validated_data.close & ...
                 validated_data.low <= validated_data.open & ...
                 validated_data.low <= validated_data.close;
    
    if ~all(ohlc_valid)
        issues_found = issues_found + sum(~ohlc_valid);
        for i = 1:length(validated_data.close)
            if ~ohlc_valid(i)
                prices = [validated_data.open(i), validated_data.close(i)];
                validated_data.high(i) = max([prices, validated_data.high(i)]);
                validated_data.low(i) = min([prices, validated_data.low(i)]);
            end
        end
    end
    
    if issues_found == 0
        fprintf('Data validation passed.\n');
    else
        fprintf('Data validation completed. %d issues corrected.\n', issues_found);
    end
end

function filtered_data = filter_data_by_index(data, valid_idx)
    fields = fieldnames(data);
    filtered_data = struct();
    
    for i = 1:length(fields)
        field_data = data.(fields{i});
        if length(field_data) == length(valid_idx)
            filtered_data.(fields{i}) = field_data(valid_idx);
        else
            filtered_data.(fields{i}) = field_data;
        end
    end
end

function filled_data = fill_missing_values(data, method)
    filled_data = data;
    
    fields_to_fill = {'open', 'high', 'low', 'close', 'adj_close', 'volume'};
    
    for f = 1:length(fields_to_fill)
        field_name = fields_to_fill{f};
        if isfield(data, field_name)
            values = data.(field_name);
            
            switch lower(method)
                case 'forward'
                    % Forward fill
                    for i = 2:length(values)
                        if isnan(values(i)) || values(i) == 0
                            values(i) = values(i-1);
                        end
                    end
                    
                case 'backward'
                    % Backward fill
                    for i = length(values)-1:-1:1
                        if isnan(values(i)) || values(i) == 0
                            values(i) = values(i+1);
                        end
                    end
                    
                case 'linear'
                    % Linear interpolation
                    valid_idx = ~isnan(values) & values > 0;
                    if sum(valid_idx) > 1
                        values = interp1(find(valid_idx), values(valid_idx), 1:length(values), 'linear', 'extrap');
                    end
            end
            
            filled_data.(field_name) = values;
        end
    end
end

function enhanced_data = add_derived_fields(data)
    % Add derived fields for analysis
    
    enhanced_data = data;
    
    % Daily returns
    enhanced_data.returns = [NaN; diff(log(data.close))];
    
    % Price changes
    enhanced_data.price_change = [NaN; diff(data.close)];
    enhanced_data.price_change_pct = enhanced_data.price_change ./ [NaN; data.close(1:end-1)] * 100;
    
    % Typical price
    enhanced_data.typical_price = (data.high + data.low + data.close) / 3;
    
    # True range for volatility calculation
    if length(data.close) > 1
        tr1 = data.high - data.low;
        tr2 = abs(data.high - [NaN; data.close(1:end-1)]);
        tr3 = abs(data.low - [NaN; data.close(1:end-1)]);
        enhanced_data.true_range = max([tr1, tr2, tr3], [], 2);
    else
        enhanced_data.true_range = data.high - data.low;
    end
    
    % Intraday range
    enhanced_data.intraday_range = (data.high - data.low) ./ data.close * 100;
    
    fprintf('Added derived fields: returns, price_change, typical_price, true_range.\n');
end

function demo_data_loading()
    % Demonstrate data loading capabilities
    
    fprintf('\n--- Data Loading Demonstration ---\n');
    
    % Generate multiple synthetic stocks
    stocks = {'AAPL', 'GOOGL', 'MSFT', 'TSLA'};
    stock_data = cell(length(stocks), 1);
    
    for i = 1:length(stocks)
        fprintf('Generating data for %s...\n', stocks{i});
        stock_data{i} = generate_synthetic_stock_data('days', 252, ...
            'initial_price', 50 + 100*rand(), ...
            'return', 0.05 + 0.15*rand(), ...
            'volatility', 0.15 + 0.25*rand());
    end
    
    % Plot all stocks
    figure('Position', [100, 100, 1400, 800]);
    
    subplot(2, 2, 1);
    colors = {'b', 'r', 'g', 'm'};
    for i = 1:length(stocks)
        plot(stock_data{i}.close, colors{i}, 'LineWidth', 1.5); hold on;
    end
    legend(stocks, 'Location', 'best');
    title('Stock Price Comparison');
    xlabel('Trading Days'); ylabel('Price ($)'); grid on;
    
    subplot(2, 2, 2);
    for i = 1:length(stocks)
        returns = stock_data{i}.returns(2:end) * 100;
        plot(returns, colors{i}, 'LineWidth', 1); hold on;
    end
    legend(stocks, 'Location', 'best');
    title('Daily Returns Comparison');
    xlabel('Trading Days'); ylabel('Return (%)'); grid on;
    
    subplot(2, 2, 3);
    volumes = zeros(length(stocks), 1);
    volatilities = zeros(length(stocks), 1);
    for i = 1:length(stocks)
        volumes(i) = mean(stock_data{i}.volume);
        volatilities(i) = std(stock_data{i}.returns(2:end)) * sqrt(252) * 100;
    end
    
    [ax, h1, h2] = plotyy(1:length(stocks), volumes/1e6, 1:length(stocks), volatilities);
    set(h1, 'LineStyle', 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    set(h2, 'LineStyle', 's-', 'LineWidth', 2, 'MarkerSize', 8);
    set(ax(1), 'XTick', 1:length(stocks), 'XTickLabel', stocks);
    set(ax(2), 'XTick', 1:length(stocks), 'XTickLabel', stocks);
    ylabel(ax(1), 'Avg Volume (M)');
    ylabel(ax(2), 'Volatility (%)');
    title('Volume vs Volatility');
    grid on;
    
    subplot(2, 2, 4);
    total_returns = zeros(length(stocks), 1);
    for i = 1:length(stocks)
        total_returns(i) = (stock_data{i}.close(end) / stock_data{i}.close(1) - 1) * 100;
    end
    
    bar(total_returns);
    set(gca, 'XTickLabel', stocks);
    title('Total Returns');
    ylabel('Return (%)'); grid on;
    
    sgtitle('Stock Data Loading Demonstration');
    
    fprintf('Data loading demonstration complete.\n');
end

function validated_data = validate_stock_data(data)
    validated_data = data;
    issues_found = 0;
    
    if any(data.open <= 0) || any(data.high <= 0) || any(data.low <= 0) || any(data.close <= 0)
        valid_idx = data.open > 0 & data.high > 0 & data.low > 0 & data.close > 0;
        validated_data = filter_data_by_index(data, valid_idx);
        issues_found = issues_found + sum(~valid_idx);
    end
    
    ohlc_valid = validated_data.high >= validated_data.low & ...
                 validated_data.high >= validated_data.open & ...
                 validated_data.high >= validated_data.close & ...
                 validated_data.low <= validated_data.open & ...
                 validated_data.low <= validated_data.close;
    
    if ~all(ohlc_valid)
        issues_found = issues_found + sum(~ohlc_valid);
        for i = 1:length(validated_data.close)
            if ~ohlc_valid(i)
                prices = [validated_data.open(i), validated_data.close(i)];
                validated_data.high(i) = max([prices, validated_data.high(i)]);
                validated_data.low(i) = min([prices, validated_data.low(i)]);
            end
        end
    end
    
    if issues_found == 0
        fprintf('Data validation passed.\n');
    else
        fprintf('Data validation completed. %d issues corrected.\n', issues_found);
    end
end

function filtered_data = filter_data_by_index(data, valid_idx)
    fields = fieldnames(data);
    filtered_data = struct();
    
    for i = 1:length(fields)
        field_data = data.(fields{i});
        if length(field_data) == length(valid_idx)
            filtered_data.(fields{i}) = field_data(valid_idx);
        else
            filtered_data.(fields{i}) = field_data;
        end
    end
end

function filled_data = fill_missing_values(data, method)
    filled_data = data;
    fields_to_fill = {'open', 'high', 'low', 'close', 'adj_close', 'volume'};
    
    for f = 1:length(fields_to_fill)
        field_name = fields_to_fill{f};
        if isfield(data, field_name)
            values = data.(field_name);
            
            switch lower(method)
                case 'forward'
                    for i = 2:length(values)
                        if isnan(values(i)) || values(i) == 0
                            values(i) = values(i-1);
                        end
                    end
                case 'linear'
                    valid_idx = ~isnan(values) & values > 0;
                    if sum(valid_idx) > 1
                        values = interp1(find(valid_idx), values(valid_idx), 1:length(values), 'linear', 'extrap');
                    end
            end
            
            filled_data.(field_name) = values;
        end
    end
end

function enhanced_data = add_derived_fields(data)
    enhanced_data = data;
    
    enhanced_data.returns = [NaN; diff(log(data.close))];
    enhanced_data.price_change = [NaN; diff(data.close)];
    enhanced_data.price_change_pct = enhanced_data.price_change ./ [NaN; data.close(1:end-1)] * 100;
    enhanced_data.typical_price = (data.high + data.low + data.close) / 3;
    
    if length(data.close) > 1
        tr1 = data.high - data.low;
        tr2 = abs(data.high - [NaN; data.close(1:end-1)]);
        tr3 = abs(data.low - [NaN; data.close(1:end-1)]);
        enhanced_data.true_range = max([tr1, tr2, tr3], [], 2);
    else
        enhanced_data.true_range = data.high - data.low;
    end
    
    enhanced_data.intraday_range = (data.high - data.low) ./ data.close * 100;
end