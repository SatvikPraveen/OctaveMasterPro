% Location: mini_projects/stock_market_analysis/price_analysis.m
% Price Analysis and Volatility Calculations

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

function grouped_levels = group_nearby_levels(levels, threshold)
    % Group price levels that are close together
    
    if isempty(levels)
        grouped_levels = [];
        return;
    end
    
    sorted_levels = sort(levels);
    grouped_levels = [];
    current_group = [sorted_levels(1)];
    
    for i = 2:length(sorted_levels)
        if abs(sorted_levels(i) - mean(current_group)) / mean(current_group) < threshold
            current_group = [current_group, sorted_levels(i)];
        else
            grouped_levels = [grouped_levels, mean(current_group)];
            current_group = [sorted_levels(i)];
        end
    end
    
    # Add final group
    grouped_levels = [grouped_levels, mean(current_group)];
end

function [drawdown, max_dd, dd_duration] = calculate_drawdown(prices)
    % Calculate drawdown analysis
    
    n = length(prices);
    cumulative_max = NaN(n, 1);
    drawdown = NaN(n, 1);
    
    cumulative_max(1) = prices(1);
    drawdown(1) = 0;
    
    for i = 2:n
        cumulative_max(i) = max(cumulative_max(i-1), prices(i));
        drawdown(i) = (prices(i) - cumulative_max(i)) / cumulative_max(i);
    end
    
    max_dd = min(drawdown);
    
    # Find drawdown duration
    in_drawdown = false;
    dd_start = 1;
    dd_durations = [];
    
    for i = 1:n
        if drawdown(i) < -0.001 && ~in_drawdown % Start of drawdown
            in_drawdown = true;
            dd_start = i;
        elseif drawdown(i) >= -0.001 && in_drawdown % End of drawdown
            in_drawdown = false;
            dd_durations = [dd_durations, i - dd_start];
        end
    end
    
    if isempty(dd_durations)
        dd_duration = 0;
    else
        dd_duration = max(dd_durations);
    end
end

function returns_stats = analyze_returns(returns)
    % Comprehensive returns analysis
    
    clean_returns = returns(~isnan(returns));
    
    returns_stats.mean_daily = mean(clean_returns);
    returns_stats.std_daily = std(clean_returns);
    returns_stats.mean_annual = returns_stats.mean_daily * 252;
    returns_stats.std_annual = returns_stats.std_daily * sqrt(252);
    
    # Sharpe ratio (assuming risk-free rate = 0)
    returns_stats.sharpe_ratio = returns_stats.mean_annual / returns_stats.std_annual;
    
    # Higher moments
    returns_stats.skewness = calculate_skewness(clean_returns);
    returns_stats.kurtosis = calculate_kurtosis(clean_returns);
    
    # Value at Risk (5%)
    returns_stats.var_5 = quantile(clean_returns, 0.05);
    
    # Maximum and minimum returns
    returns_stats.max_return = max(clean_returns);
    returns_stats.min_return = min(clean_returns);
    
    # Positive/negative days
    returns_stats.positive_days = sum(clean_returns > 0) / length(clean_returns);
    returns_stats.negative_days = sum(clean_returns < 0) / length(clean_returns);
end

function skew = calculate_skewness(data)
    mu = mean(data);
    sigma = std(data);
    skew = mean(((data - mu) / sigma).^3);
end

function kurt = calculate_kurtosis(data)
    mu = mean(data);
    sigma = std(data);
    kurt = mean(((data - mu) / sigma).^4) - 3;
end

function q = quantile(data, p)
    % Calculate quantile
    sorted_data = sort(data);
    n = length(sorted_data);
    index = p * (n - 1) + 1;
    
    if index == round(index)
        q = sorted_data(round(index));
    else
        lower = floor(index);
        upper = ceil(index);
        weight = index - lower;
        q = sorted_data(lower) * (1 - weight) + sorted_data(upper) * weight;
    end
end

function demo_price_analysis()
    % Demonstrate comprehensive price analysis
    
    fprintf('\n--- Price Analysis Demonstration ---\n');
    
    # Load data
    data = load_stock_data('');
    prices = data.close;
    returns = data.returns;
    
    fprintf('Performing comprehensive price analysis...\n');
    
    # Calculate analysis components
    [trend, trend_strength] = trend_analysis(prices, 20);
    volatility = calculate_volatility(returns, 20, 'standard');
    exp_volatility = calculate_volatility(returns, 20, 'exponential');
    gk_vol = garman_klass_volatility(data.high, data.low, data.open, data.close, 20);
    
    [drawdown, max_dd, dd_duration] = calculate_drawdown(prices);
    returns_stats = analyze_returns(returns);
    sr_levels = find_support_resistance(prices, 15, 0.02);
    
    # Visualize results
    figure('Position', [50, 50, 1400, 1200]);
    
    # Price with trend
    subplot(3, 3, 1);
    plot(prices, 'k', 'LineWidth', 1.5); hold on;
    
    # Highlight trend periods
    strong_uptrend = trend > 0 & trend_strength > 0.7;
    strong_downtrend = trend < 0 & trend_strength > 0.7;
    
    if any(strong_uptrend)
        plot(find(strong_uptrend), prices(strong_uptrend), 'g.', 'MarkerSize', 8);
    end
    if any(strong_downtrend)
        plot(find(strong_downtrend), prices(strong_downtrend), 'r.', 'MarkerSize', 8);
    end
    
    # Add support/resistance lines
    for level = sr_levels.support_levels
        plot([1, length(prices)], [level, level], 'g--', 'LineWidth', 1);
    end
    for level = sr_levels.resistance_levels
        plot([1, length(prices)], [level, level], 'r--', 'LineWidth', 1);
    end
    
    title('Price Analysis with S/R Levels');
    ylabel('Price ($)'); grid on;
    
    # Volatility comparison
    subplot(3, 3, 2);
    plot(volatility * 100, 'b', 'LineWidth', 1.5); hold on;
    plot(exp_volatility * 100, 'r', 'LineWidth', 1);
    plot(gk_vol * 100, 'g', 'LineWidth', 1);
    legend('Standard', 'Exponential', 'Garman-Klass', 'Location', 'best');
    title('Volatility Comparison');
    ylabel('Annualized Vol (%)'); grid on;
    
    # Returns distribution
    subplot(3, 3, 3);
    clean_returns = returns(~isnan(returns)) * 100;
    hist(clean_returns, 30);
    title('Daily Returns Distribution');
    xlabel('Returns (%)'); ylabel('Frequency'); grid on;
    
    # Drawdown analysis
    subplot(3, 3, 4);
    plot(drawdown * 100, 'r', 'LineWidth', 1.5);
    title(sprintf('Drawdown Analysis (Max: %.1f%%)', max_dd * 100));
    ylabel('Drawdown (%)'); grid on;
    
    # Trend analysis
    subplot(3, 3, 5);
    [ax, h1, h2] = plotyy(1:length(trend), trend, 1:length(trend_strength), trend_strength);
    set(h1, 'LineWidth', 1.5, 'Color', 'blue');
    set(h2, 'LineWidth', 1.5, 'Color', 'red');
    ylabel(ax(1), 'Trend Slope');
    ylabel(ax(2), 'Trend Strength (R²)');
    title('Trend Analysis');
    grid on;
    
    # Risk metrics summary
    subplot(3, 3, 6);
    metrics = {'Ann. Return', 'Ann. Vol', 'Sharpe', 'Max DD', 'VaR 5%'};
    values = [returns_stats.mean_annual*100, returns_stats.std_annual*100, ...
              returns_stats.sharpe_ratio, max_dd*100, returns_stats.var_5*100];
    
    bar(values);
    set(gca, 'XTickLabel', metrics);
    title('Risk Metrics Summary');
    ylabel('Values'); grid on;
    
    # Rolling correlations (if multiple assets)
    subplot(3, 3, 7);
    rolling_vol = volatility * 100;
    plot(rolling_vol, 'purple', 'LineWidth', 1.5);
    title('Rolling 20-Day Volatility');
    ylabel('Volatility (%)'); grid on;
    
    # Price momentum
    subplot(3, 3, 8);
    momentum_5 = [NaN(5,1); prices(6:end) - prices(1:end-5)];
    momentum_20 = [NaN(20,1); prices(21:end) - prices(1:end-20)];
    
    plot(momentum_5, 'b', 'LineWidth', 1); hold on;
    plot(momentum_20, 'r', 'LineWidth', 1.5);
    legend('5-day', '20-day', 'Location', 'best');
    title('Price Momentum');
    ylabel('Price Change ($)'); grid on;
    
    # Return autocorrelation
    subplot(3, 3, 9);
    lags = 1:20;
    autocorr_values = NaN(length(lags), 1);
    clean_returns = returns(~isnan(returns));
    
    for lag = lags
        if length(clean_returns) > lag
            autocorr_values(lag) = correlation_coefficient(clean_returns(1:end-lag), clean_returns(1+lag:end));
        end
    end
    
    bar(lags, autocorr_values);
    title('Return Autocorrelation');
    xlabel('Lag (days)'); ylabel('Correlation'); grid on;
    
    sgtitle('Comprehensive Price Analysis');
    
    # Print summary statistics
    fprintf('\nPrice Analysis Summary:\n');
    fprintf('======================\n');
    fprintf('Annual Return: %.2f%%\n', returns_stats.mean_annual*100);
    fprintf('Annual Volatility: %.2f%%\n', returns_stats.std_annual*100);
    fprintf('Sharpe Ratio: %.2f\n', returns_stats.sharpe_ratio);
    fprintf('Maximum Drawdown: %.2f%%\n', max_dd*100);
    fprintf('Value at Risk (5%%): %.2f%%\n', returns_stats.var_5*100);
    fprintf('Skewness: %.2f\n', returns_stats.skewness);
    fprintf('Kurtosis: %.2f\n', returns_stats.kurtosis);
    fprintf('Positive Trading Days: %.1f%%\n', returns_stats.positive_days*100);
end

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

function regime_analysis(prices, returns)
    % Market regime analysis (bull/bear markets)
    
    fprintf('\n--- Market Regime Analysis ---\n');
    
    # Simple regime detection based on moving averages
    sma_short = simple_moving_average(prices, 50);
    sma_long = simple_moving_average(prices, 200);
    
    # Bull market: short MA > long MA
    bull_market = sma_short > sma_long;
    
    # Find regime changes
    regime_changes = [false; diff(bull_market) ~= 0];
    change_indices = find(regime_changes);
    
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 1, 1);
    plot(prices, 'k', 'LineWidth', 1.5); hold on;
    plot(sma_short, 'b', 'LineWidth', 1);
    plot(sma_long, 'r', 'LineWidth', 1);
    
    # Color background by regime
    y_limits = ylim;
    for i = 1:length(change_indices)-1
        start_idx = change_indices(i);
        end_idx = change_indices(i+1) - 1;
        
        if bull_market(start_idx)
            fill([start_idx, end_idx, end_idx, start_idx], ...
                 [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
                 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        else
            fill([start_idx, end_idx, end_idx, start_idx], ...
                 [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
                 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        end
    end
    
    legend('Price', 'SMA-50', 'SMA-200', 'Location', 'best');
    title('Market Regimes (Green=Bull, Red=Bear)');
    ylabel('Price ($)'); grid on;
    
    subplot(2, 1, 2);
    cumulative_returns = cumprod(1 + returns(~isnan(returns))) - 1;
    plot(cumulative_returns * 100, 'k', 'LineWidth', 2);
    title('Cumulative Returns');
    xlabel('Trading Days'); ylabel('Cumulative Return (%)'); grid on;
    
    # Calculate regime statistics
    bull_periods = sum(bull_market(~isnan(bull_market)));
    total_periods = sum(~isnan(bull_market));
    bull_percentage = bull_periods / total_periods * 100;
    
    fprintf('Market was in bull regime %.1f%% of the time.\n', bull_percentage);
    fprintf('Number of regime changes: %d\n', length(change_indices));
end

function seasonal_analysis(prices, dates)
    % Analyze seasonal patterns in stock returns
    
    fprintf('\n--- Seasonal Analysis ---\n');
    
    # Extract month information (simplified)
    months = mod(floor(dates/30), 12) + 1; % Approximate month extraction
    
    monthly_returns = NaN(12, 1);
    monthly_volatility = NaN(12, 1);
    
    returns = [NaN; diff(log(prices))];
    
    for month = 1:12
        month_mask = months == month;
        if sum(month_mask) > 0
            month_returns = returns(month_mask);
            month_returns = month_returns(~isnan(month_returns));
            
            if length(month_returns) > 1
                monthly_returns(month) = mean(month_returns) * 252 * 100; % Annualized %
                monthly_volatility(month) = std(month_returns) * sqrt(252) * 100;
            end
        end
    end
    
    # Plot seasonal patterns
    figure('Position', [200, 200, 1200, 600]);
    
    subplot(1, 2, 1);
    bar(monthly_returns);
    month_names = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', ...
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
    set(gca, 'XTickLabel', month_names);
    title('Average Monthly Returns');
    ylabel('Annualized Return (%)'); grid on;
    
    subplot(1, 2, 2);
    bar(monthly_volatility);
    set(gca, 'XTickLabel', month_names);
    title('Average Monthly Volatility');
    ylabel('Annualized Volatility (%)'); grid on;
    
    sgtitle('Seasonal Analysis');
    
    fprintf('Seasonal analysis complete.\n');
end