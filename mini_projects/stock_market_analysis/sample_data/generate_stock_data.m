% Location: mini_projects/stock_market_analysis/sample_data/generate_stock_data.m
% Generate Realistic Stock Market Data

function generate_stock_data()
    % Generate comprehensive stock market data for demonstrations
    
    fprintf('Generating stock market data...\n');
    
    % Create data directory if it doesn't exist
    data_dir = fileparts(mfilename('fullpath'));
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    % Define stock parameters
    stocks = {
        struct('symbol', 'AAPL', 'name', 'Apple Inc.', 'initial_price', 320, 'annual_return', 0.15, 'volatility', 0.25);
        struct('symbol', 'GOOGL', 'name', 'Alphabet Inc.', 'initial_price', 1800, 'annual_return', 0.12, 'volatility', 0.22);
        struct('symbol', 'MSFT', 'name', 'Microsoft Corp.', 'initial_price', 220, 'annual_return', 0.18, 'volatility', 0.23);
        struct('symbol', 'TSLA', 'name', 'Tesla Inc.', 'initial_price', 400, 'annual_return', 0.25, 'volatility', 0.45);
        struct('symbol', 'SPY', 'name', 'SPDR S&P 500 ETF', 'initial_price', 340, 'annual_return', 0.08, 'volatility', 0.18);
    };
    
    % Generate individual stock data
    for i = 1:length(stocks)
        generate_individual_stock(data_dir, stocks{i});
    end
    
    % Generate portfolio data
    generate_portfolio_data(data_dir, stocks);
    
    % Generate market events data
    generate_market_events_data(data_dir);
    
    fprintf('Stock market data generation complete!\n');
    fprintf('Generated files in: %s\n', data_dir);
end

function generate_individual_stock(data_dir, stock_params)
    % Generate realistic individual stock data
    
    fprintf('  Generating %s data...\n', stock_params.symbol);
    
    % Date range: Jan 1, 2020 to Dec 31, 2023
    start_date = datenum('2020-01-01');
    end_date = datenum('2023-12-31');
    dates = start_date:end_date; % Daily data
    num_days = length(dates);
    
    % Convert to daily parameters
    daily_return = stock_params.annual_return / 252;
    daily_vol = stock_params.volatility / sqrt(252);
    
    % Set random seed for reproducibility
    rand('state', sum(double(stock_params.symbol))); % Unique seed per stock
    randn('state', sum(double(stock_params.symbol)));
    
    % Generate price series using geometric Brownian motion with market effects
    log_prices = zeros(num_days, 1);
    log_prices(1) = log(stock_params.initial_price);
    
    for i = 2:num_days
        % Base return with drift
        base_return = daily_return + daily_vol * randn();
        
        # Add market regime effects
        market_effect = get_market_effect(dates(i), stock_params.symbol);
        
        # Add volatility clustering (GARCH-like effect)
        if i > 10
            recent_vol = std(diff(log_prices(max(1,i-10):i-1)));
            vol_adjustment = 0.1 * (recent_vol - daily_vol);
        else
            vol_adjustment = 0;
        end
        
        # Add momentum effect
        if i > 5
            momentum = mean(diff(log_prices(i-5:i-1))) * 0.1;
        else
            momentum = 0;
        end
        
        # Combined return
        total_return = base_return + market_effect + vol_adjustment + momentum;
        log_prices(i) = log_prices(i-1) + total_return;
    end
    
    # Convert to actual prices
    close_prices = exp(log_prices);
    
    # Generate OHLV data
    open_prices = zeros(size(close_prices));
    high_prices = zeros(size(close_prices));
    low_prices = zeros(size(close_prices));
    volumes = zeros(size(close_prices));
    
    open_prices(1) = close_prices(1);
    
    for i = 2:num_days
        # Opening price (gap from previous close)
        gap_size = 0.002 * randn(); # 0.2% average gap
        open_prices(i) = close_prices(i-1) * (1 + gap_size);
        
        # Intraday range based on volatility
        daily_range = abs(close_prices(i) - open_prices(i)) + ...
                     daily_vol * close_prices(i) * abs(randn());
        
        # High and low prices
        high_prices(i) = max(open_prices(i), close_prices(i)) + daily_range * rand();
        low_prices(i) = min(open_prices(i), close_prices(i)) - daily_range * rand();
        
        # Ensure price consistency
        high_prices(i) = max(high_prices(i), max(open_prices(i), close_prices(i)));
        low_prices(i) = min(low_prices(i), min(open_prices(i), close_prices(i)));
        
        # Volume generation (negatively correlated with price, positively with volatility)
        price_change = abs(close_prices(i) - close_prices(i-1)) / close_prices(i-1);
        base_volume = 50000000 * (1 + price_change * 5); # Higher volume on big moves
        volume_noise = 0.5 * randn();
        volumes(i) = max(1000000, base_volume * (1 + volume_noise)); # Minimum 1M volume
    end
    
    # Handle first day
    high_prices(1) = close_prices(1) * (1 + 0.01*abs(randn()));
    low_prices(1) = close_prices(1) * (1 - 0.01*abs(randn()));
    volumes(1) = 30000000;
    
    # Adjusted close (same as close for simplicity)
    adj_close = close_prices;
    
    # Create date strings
    date_strings = cell(num_days, 1);
    for i = 1:num_days
        date_strings{i} = datestr(dates(i), 'yyyy-mm-dd');
    end
    
    # Create CSV content
    csv_filename = fullfile(data_dir, [stock_params.symbol, '_2020_2023.csv']);
    fid = fopen(csv_filename, 'w');
    
    # Write header
    fprintf(fid, 'Date,Open,High,Low,Close,Volume,Adj_Close\n');
    
    # Write data
    for i = 1:num_days
        fprintf(fid, '%s,%.2f,%.2f,%.2f,%.2f,%.0f,%.2f\n', ...
               date_strings{i}, open_prices(i), high_prices(i), low_prices(i), ...
               close_prices(i), volumes(i), adj_close(i));
    end
    
    fclose(fid);
end

function market_effect = get_market_effect(current_date, symbol)
    % Get market regime effects for specific dates
    
    market_effect = 0;
    
    # COVID crash (March 2020)
    covid_crash_start = datenum('2020-03-01');
    covid_crash_end = datenum('2020-04-01');
    
    if current_date >= covid_crash_start && current_date <= covid_crash_end
        crash_intensity = (current_date - covid_crash_start) / (covid_crash_end - covid_crash_start);
        # Different stocks affected differently
        if strcmp(symbol, 'TSLA')
            market_effect = -0.03 * sin(pi * crash_intensity); # High volatility stock
        elseif strcmp(symbol, 'SPY')
            market_effect = -0.02 * sin(pi * crash_intensity); # Market index
        else
            market_effect = -0.025 * sin(pi * crash_intensity); # Other stocks
        end
    end
    
    # Tech rally (2020-2021)
    tech_rally_start = datenum('2020-05-01');
    tech_rally_end = datenum('2021-12-31');
    
    if current_date >= tech_rally_start && current_date <= tech_rally_end
        if strcmp(symbol, 'AAPL') || strcmp(symbol, 'GOOGL') || strcmp(symbol, 'MSFT')
            market_effect = market_effect + 0.0005; # Tech boost
        elseif strcmp(symbol, 'TSLA')
            market_effect = market_effect + 0.001; # Extra Tesla boost
        end
    end
    
    # Interest rate concerns (2022)
    rate_concerns_start = datenum('2022-01-01');
    rate_concerns_end = datenum('2022-12-31');
    
    if current_date >= rate_concerns_start && current_date <= rate_concerns_end
        # Growth stocks more affected
        if strcmp(symbol, 'TSLA')
            market_effect = market_effect - 0.0008;
        elseif strcmp(symbol, 'GOOGL') || strcmp(symbol, 'AAPL')
            market_effect = market_effect - 0.0003;
        end
    end
    
    # Add some randomness to market effects
    market_effect = market_effect * (0.8 + 0.4*rand());
end

function generate_portfolio_data(data_dir, stocks)
    % Generate multi-asset portfolio data
    
    fprintf('  Generating portfolio data...\n');
    
    # Read generated stock data
    portfolio_data = struct();
    
    for i = 1:length(stocks)
        filename = fullfile(data_dir, [stocks{i}.symbol, '_2020_2023.csv']);
        if exist(filename, 'file')
            # Read the CSV file (simplified parsing)
            fid = fopen(filename, 'r');
            header = fgetl(fid); # Skip header
            
            dates = {};
            prices = [];
            line_count = 0;
            
            while ~feof(fid)
                line = fgetl(fid);
                if ischar(line) && ~isempty(line)
                    parts = strsplit(line, ',');
                    if length(parts) >= 5
                        line_count = line_count + 1;
                        dates{line_count} = parts{1};
                        prices(line_count) = str2double(parts{5}); # Close price
                    end
                end
            end
            fclose(fid);
            
            portfolio_data.(stocks{i}.symbol) = struct('dates', {dates}, 'prices', prices);
        end
    end
    
    # Create portfolio allocation CSV
    csv_filename = fullfile(data_dir, 'portfolio_sample.csv');
    fid = fopen(csv_filename, 'w');
    
    # Write header
    fprintf(fid, 'Date');
    for i = 1:length(stocks)
        fprintf(fid, ',%s', stocks{i}.symbol);
    end
    fprintf(fid, ',Portfolio_Value\n');
    
    # Portfolio weights (strategic allocation)
    weights = [0.25, 0.20, 0.25, 0.15, 0.15]; # AAPL, GOOGL, MSFT, TSLA, SPY
    initial_investment = 100000; # $100k portfolio
    
    # Calculate portfolio value over time
    if isfield(portfolio_data, 'AAPL') && ~isempty(portfolio_data.AAPL.dates)
        num_days = length(portfolio_data.AAPL.dates);
        
        for day = 1:num_days
            fprintf(fid, '%s', portfolio_data.AAPL.dates{day});
            
            portfolio_value = 0;
            for i = 1:length(stocks)
                symbol = stocks{i}.symbol;
                if isfield(portfolio_data, symbol) && day <= length(portfolio_data.(symbol).prices)
                    stock_price = portfolio_data.(symbol).prices(day);
                    initial_stock_price = portfolio_data.(symbol).prices(1);
                    stock_return = stock_price / initial_stock_price;
                    stock_contribution = weights(i) * initial_investment * stock_return;
                    portfolio_value = portfolio_value + stock_contribution;
                    fprintf(fid, ',%.2f', stock_price);
                else
                    fprintf(fid, ',');
                end
            end
            
            fprintf(fid, ',%.2f\n', portfolio_value);
        end
    end
    
    fclose(fid);
end

function generate_market_events_data(data_dir)
    % Generate market events and sentiment data
    
    fprintf('  Generating market events data...\n');
    
    # Market events CSV
    events_filename = fullfile(data_dir, 'market_events.csv');
    fid = fopen(events_filename, 'w');
    
    fprintf(fid, 'Date,Event_Type,Description,Market_Impact\n');
    
    # Define major market events
    events = {
        {'2020-03-11', 'Health_Crisis', 'WHO declares COVID-19 pandemic', 'Negative'};
        {'2020-03-23', 'Policy', 'Federal Reserve announces unlimited QE', 'Positive'};
        {'2020-11-09', 'Medical', 'Pfizer announces 90%% effective vaccine', 'Positive'};
        {'2021-01-06', 'Political', 'US Capitol riots', 'Negative'};
        {'2021-03-11', 'Policy', 'American Rescue Plan signed', 'Positive'};
        {'2021-11-10', 'Economic', 'Inflation reaches 6.2%%, highest since 1990', 'Negative'};
        {'2022-02-24', 'Geopolitical', 'Russia invades Ukraine', 'Negative'};
        {'2022-03-16', 'Policy', 'Fed raises rates by 0.25%% (first since 2018)', 'Negative'};
        {'2022-06-15', 'Policy', 'Fed raises rates by 0.75%% (largest since 1994)', 'Negative'};
        {'2022-10-13', 'Economic', 'UK budget crisis and pound crash', 'Negative'};
        {'2023-03-10', 'Financial', 'Silicon Valley Bank failure', 'Negative'};
        {'2023-05-03', 'Policy', 'Fed raises rates to 5.25%%, signals pause', 'Mixed'};
        {'2023-11-01', 'Technology', 'ChatGPT anniversary drives AI stock rally', 'Positive'};
    };
    
    for i = 1:length(events)
        event = events{i};
        fprintf(fid, '%s,%s,"%s",%s\n', event{1}, event{2}, event{3}, event{4});
    end
    
    fclose(fid);
    
    # Economic indicators CSV
    indicators_filename = fullfile(data_dir, 'economic_indicators.csv');
    fid = fopen(indicators_filename, 'w');
    
    fprintf(fid, 'Date,GDP_Growth,Unemployment_Rate,Inflation_Rate,Fed_Funds_Rate\n');
    
    # Generate quarterly economic data
    start_date = datenum('2020-01-01');
    end_date = datenum('2023-12-31');
    
    # Quarterly dates
    quarters = [];
    current_date = start_date;
    while current_date <= end_date
        quarters = [quarters, current_date];
        # Move to next quarter
        [year, month, day] = datevec(current_date);
        if month <= 3
            next_quarter = datenum(year, 4, 1);
        elseif month <= 6
            next_quarter = datenum(year, 7, 1);
        elseif month <= 9
            next_quarter = datenum(year, 10, 1);
        else
            next_quarter = datenum(year + 1, 1, 1);
        end
        current_date = next_quarter;
    end
    
    # Generate economic indicators with realistic patterns
    for i = 1:length(quarters)
        date_str = datestr(quarters(i), 'yyyy-mm-dd');
        
        # GDP Growth (annualized quarterly rate)
        if quarters(i) < datenum('2020-04-01')
            gdp = 2.0 + randn() * 0.5; # Pre-pandemic
        elseif quarters(i) < datenum('2020-07-01')
            gdp = -8.0 + randn() * 2.0; # Pandemic crash
        elseif quarters(i) < datenum('2022-01-01')
            gdp = 4.0 + randn() * 1.0; # Recovery
        else
            gdp = 1.5 + randn() * 0.8; # Slower growth
        end
        
        # Unemployment Rate
        if quarters(i) < datenum('2020-04-01')
            unemployment = 3.5 + randn() * 0.2; # Low pre-pandemic
        elseif quarters(i) < datenum('2021-01-01')
            unemployment = 10.0 - 5.0 * (quarters(i) - datenum('2020-04-01')) / 365; # Recovery
        else
            unemployment = max(3.0, 6.0 - 2.0 * (quarters(i) - datenum('2021-01-01')) / 365); # Continued improvement
        end
        
        # Inflation Rate (CPI YoY)
        if quarters(i) < datenum('2021-01-01')
            inflation = 1.0 + randn() * 0.3; # Low inflation
        elseif quarters(i) < datenum('2022-07-01')
            inflation = 2.0 + 4.0 * (quarters(i) - datenum('2021-01-01')) / 365; # Rising inflation
        else
            inflation = max(2.0, 7.0 - 3.0 * (quarters(i) - datenum('2022-07-01')) / 365); # Declining from peak
        end
        
        # Fed Funds Rate
        if quarters(i) < datenum('2022-03-01')
            fed_rate = 0.25; # Near zero
        else
            # Rising rates
            months_since_march_2022 = (quarters(i) - datenum('2022-03-01')) / 30;
            fed_rate = min(5.25, 0.25 + months_since_march_2022 * 0.3);
        end
        
        fprintf(fid, '%s,%.2f,%.2f,%.2f,%.2f\n', date_str, gdp, unemployment, inflation, fed_rate);
    end
    
    fclose(fid);
end