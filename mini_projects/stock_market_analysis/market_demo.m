% Location: mini_projects/stock_market_analysis/market_demo.m
% Main Stock Market Analysis Demonstration

function market_demo()
    clear; clc; close all;
    
    fprintf('====================================================\n');
    fprintf('      STOCK MARKET ANALYSIS DEMONSTRATION          \n');
    fprintf('====================================================\n\n');
    
    try
        while true
            fprintf('\nSelect a demonstration:\n');
            fprintf('1. Data Loading & Preprocessing\n');
            fprintf('2. Technical Indicators\n');
            fprintf('3. Price & Trend Analysis\n');
            fprintf('4. Portfolio Optimization\n');
            fprintf('5. Risk Analysis\n');
            fprintf('6. Complete Trading Strategy\n');
            fprintf('0. Exit\n');
            
            choice = input('Enter your choice (0-6): ');
            
            switch choice
                case 0
                    fprintf('\nExiting Market Analysis Demo. Goodbye!\n');
                    break;
                case 1
                    data_loading_demo();
                case 2
                    technical_indicators_demo();
                case 3
                    price_analysis_demo();
                case 4
                    portfolio_optimization_demo();
                case 5
                    risk_analysis_demo();
                case 6
                    complete_strategy_demo();
                otherwise
                    fprintf('Invalid choice. Please select 0-6.\n');
            end
            
            if choice ~= 0
                input('\nPress Enter to continue...');
            end
        end
        
    catch err
        fprintf('Error in market_demo: %s\n', err.message);
    end
end

function data_loading_demo()
    fprintf('\n--- Data Loading Demo ---\n');
    demo_data_loading();
end

function technical_indicators_demo()
    fprintf('\n--- Technical Indicators Demo ---\n');
    demo_technical_indicators();
end

function price_analysis_demo()
    fprintf('\n--- Price Analysis Demo ---\n');
    demo_price_analysis();
end

function portfolio_optimization_demo()
    fprintf('\n--- Portfolio Optimization Demo ---\n');
    demo_portfolio_optimization();
end

function risk_analysis_demo()
    fprintf('\n--- Risk Analysis Demo ---\n');
    
    # Load sample data
    data = load_stock_data('');
    returns = data.returns(~isnan(data.returns));
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(returns);
    
    figure('Position', [100, 100, 1400, 800]);
    
    # Returns distribution
    subplot(2, 3, 1);
    hist(returns * 100, 30);
    hold on;
    plot([risk_metrics.var_95*100, risk_metrics.var_95*100], ylim, 'r--', 'LineWidth', 2);
    plot([risk_metrics.var_99*100, risk_metrics.var_99*100], ylim, 'r-', 'LineWidth', 2);
    title('Returns Distribution');
    xlabel('Daily Returns (%)'); ylabel('Frequency');
    legend('Returns', '95% VaR', '99% VaR', 'Location', 'best');
    grid on;
    
    # Rolling volatility
    subplot(2, 3, 2);
    vol_rolling = calculate_volatility(returns, 20, 'standard') * 100;
    vol_exp = calculate_volatility(returns, 20, 'exponential') * 100;
    
    plot(vol_rolling, 'b', 'LineWidth', 1.5); hold on;
    plot(vol_exp, 'r', 'LineWidth', 1.5);
    legend('Standard', 'Exponential', 'Location', 'best');
    title('Rolling Volatility');
    ylabel('Annualized Vol (%)'); grid on;
    
    # Drawdown analysis
    cumulative_returns = cumprod(1 + returns);
    [drawdown, max_dd, dd_duration] = calculate_drawdown(cumulative_returns);
    
    subplot(2, 3, 3);
    plot(drawdown * 100, 'r', 'LineWidth', 1.5);
    title(sprintf('Drawdown (Max: %.1f%%, Duration: %d days)', max_dd*100, dd_duration));
    ylabel('Drawdown (%)'); grid on;
    
    # Risk metrics summary
    subplot(2, 3, 4);
    metrics = {'Vol', 'Down Dev', 'Sortino', 'Calmar', 'Max DD'};
    values = [risk_metrics.volatility*100, risk_metrics.downside_deviation*100, ...
              risk_metrics.sortino_ratio, risk_metrics.calmar_ratio, abs(risk_metrics.max_drawdown)*100];
    
    bar(values);
    set(gca, 'XTickLabel', metrics);
    title('Risk Metrics');
    ylabel('Values'); grid on;
    
    # VaR analysis
    subplot(2, 3, 5);
    confidence_levels = [0.01, 0.05, 0.10, 0.20];
    var_values = NaN(length(confidence_levels), 1);
    
    for i = 1:length(confidence_levels)
        var_values(i) = quantile_simple(returns, confidence_levels(i)) * 100;
    end
    
    bar(var_values);
    set(gca, 'XTickLabel', {'99%', '95%', '90%', '80%'});
    title('Value at Risk');
    ylabel('Daily VaR (%)'); grid on;
    
    # Return autocorrelations
    subplot(2, 3, 6);
    lags = 1:10;
    autocorr = NaN(length(lags), 1);
    
    for lag = lags
        if length(returns) > lag
            autocorr(lag) = correlation_coefficient(returns(1:end-lag), returns(1+lag:end));
        end
    end
    
    bar(lags, autocorr);
    title('Return Autocorrelations');
    xlabel('Lag (days)'); ylabel('Correlation'); grid on;
    
    sgtitle('Comprehensive Risk Analysis');
    
    fprintf('Risk analysis complete.\n');
    display_risk_summary(risk_metrics);
end

function display_risk_summary(risk_metrics)
    fprintf('\nRisk Analysis Summary:\n');
    fprintf('=====================\n');
    fprintf('Annual Volatility: %.2f%%\n', risk_metrics.volatility*100);
    fprintf('Downside Deviation: %.2f%%\n', risk_metrics.downside_deviation*100);
    fprintf('VaR (95%%): %.2f%%\n', risk_metrics.var_95*100);
    fprintf('CVaR (95%%): %.2f%%\n', risk_metrics.cvar_95*100);
    fprintf('Maximum Drawdown: %.2f%%\n', abs(risk_metrics.max_drawdown)*100);
    fprintf('Sortino Ratio: %.3f\n', risk_metrics.sortino_ratio);
    fprintf('Calmar Ratio: %.3f\n', risk_metrics.calmar_ratio);
end

function complete_strategy_demo()
    fprintf('\n--- Complete Trading Strategy Demo ---\n');
    
    # Generate multi-asset universe
    num_assets = 5;
    num_days = 504; # 2 years of data
    
    returns_matrix = generate_correlated_returns(num_assets, num_days);
    asset_names = {'Large Cap', 'Small Cap', 'International', 'Bonds', 'Commodities'};
    
    fprintf('Implementing complete trading strategy...\n');
    
    # Step 1: Portfolio optimization
    weights_strategic = optimize_portfolio(returns_matrix, 'method', 'max_sharpe');
    
    # Step 2: Generate price series for technical analysis
    prices_matrix = NaN(num_days, num_assets);
    initial_prices = [100, 50, 80, 1000, 200];
    
    for asset = 1:num_assets
        log_prices = log(initial_prices(asset)) + cumsum(returns_matrix(:, asset));
        prices_matrix(:, asset) = exp(log_prices);
    end
    
    # Step 3: Technical analysis signals
    signals_matrix = NaN(num_days, num_assets);
    
    for asset = 1:num_assets
        prices = prices_matrix(:, asset);
        
        # Moving average crossover
        sma_fast = simple_moving_average(prices, 20);
        sma_slow = simple_moving_average(prices, 50);
        ma_signal = double(sma_fast > sma_slow);
        
        # RSI signal
        rsi = relative_strength_index(prices, 14);
        rsi_signal = double(rsi < 70 & rsi > 30); # Avoid overbought/oversold
        
        # Combined signal
        signals_matrix(:, asset) = ma_signal .* rsi_signal;
    end
    
    # Step 4: Dynamic allocation
    dynamic_weights = NaN(num_days, num_assets);
    base_weights = weights_strategic';
    
    for day = 1:num_days
        if day >= 50 # Need enough data for indicators
            signal_strength = signals_matrix(day, :);
            # Adjust weights based on signals
            adjusted_weights = base_weights .* (0.5 + 0.5 * signal_strength);
            adjusted_weights = adjusted_weights / sum(adjusted_weights);
            dynamic_weights(day, :) = adjusted_weights;
        else
            dynamic_weights(day, :) = base_weights;
        end
    end
    
    # Step 5: Backtest comparison
    static_backtest = backtest_portfolio(returns_matrix, weights_strategic, 'rebalance', 'monthly');
    
    # Dynamic backtest (simplified)
    dynamic_portfolio_value = 1;
    dynamic_values = NaN(num_days, 1);
    
    for day = 1:num_days
        if day == 1
            dynamic_values(day) = dynamic_portfolio_value;
        else
            daily_return = dynamic_weights(day-1, :) * returns_matrix(day, :)';
            dynamic_portfolio_value = dynamic_portfolio_value * (1 + daily_return);
            dynamic_values(day) = dynamic_portfolio_value;
        end
    end
    
    # Visualize complete strategy
    figure('Position', [50, 50, 1400, 1000]);
    
    # Portfolio performance comparison
    subplot(2, 3, 1);
    plot(static_backtest.portfolio_values, 'b', 'LineWidth', 2); hold on;
    plot(dynamic_values, 'r', 'LineWidth', 2);
    legend('Static Portfolio', 'Dynamic Strategy', 'Location', 'best');
    title('Strategy Performance Comparison');
    ylabel('Portfolio Value'); grid on;
    
    # Weight evolution
    subplot(2, 3, 2);
    plot(dynamic_weights);
    legend(asset_names, 'Location', 'best');
    title('Dynamic Weight Allocation');
    ylabel('Weight'); grid on;
    
    # Asset prices
    subplot(2, 3, 3);
    normalized_prices = prices_matrix ./ prices_matrix(1, :);
    plot(normalized_prices);
    legend(asset_names, 'Location', 'best');
    title('Normalized Asset Prices');
    ylabel('Normalized Price'); grid on;
    
    # Signal strength heatmap
    subplot(2, 3, 4);
    imagesc(signals_matrix');
    colorbar;
    set(gca, 'YTick', 1:num_assets, 'YTickLabel', asset_names);
    title('Signal Strength Over Time');
    xlabel('Trading Days');
    
    # Performance metrics
    subplot(2, 3, 5);
    static_return = (static_backtest.portfolio_values(end) - 1) * 100;
    dynamic_return = (dynamic_values(end) - 1) * 100;
    
    static_vol = std(diff(log(static_backtest.portfolio_values))) * sqrt(252) * 100;
    dynamic_vol = std(diff(log(dynamic_values))) * sqrt(252) * 100;
    
    metrics = {'Total Return', 'Volatility', 'Sharpe'};
    static_vals = [static_return, static_vol, static_return/static_vol];
    dynamic_vals = [dynamic_return, dynamic_vol, dynamic_return/dynamic_vol];
    
    x = 1:length(metrics);
    bar(x-0.2, static_vals, 0.4); hold on;
    bar(x+0.2, dynamic_vals, 0.4);
    
    set(gca, 'XTickLabel', metrics);
    legend('Static', 'Dynamic', 'Location', 'best');
    title('Performance Comparison');
    ylabel('Values'); grid on;
    
    # Rolling Sharpe ratios
    subplot(2, 3, 6);
    window = 60;
    static_rolling_sharpe = calculate_rolling_sharpe(diff(log(static_backtest.portfolio_values)), window);
    dynamic_rolling_sharpe = calculate_rolling_sharpe(diff(log(dynamic_values)), window);
    
    plot(static_rolling_sharpe, 'b', 'LineWidth', 1.5); hold on;
    plot(dynamic_rolling_sharpe, 'r', 'LineWidth', 1.5);
    legend('Static Portfolio', 'Dynamic Strategy', 'Location', 'best');
    title('Rolling 60-Day Sharpe Ratio');
    ylabel('Sharpe Ratio'); grid on;
    
    sgtitle('Complete Trading Strategy Analysis');
    
    fprintf('\nStrategy Results:\n');
    fprintf('Static Portfolio: %.2f%% return, %.2f%% vol\n', static_return, static_vol);
    fprintf('Dynamic Strategy: %.2f%% return, %.2f%% vol\n', dynamic_return, dynamic_vol);
    
    fprintf('Complete trading strategy demonstration finished.\n');
end

function rolling_sharpe = calculate_rolling_sharpe(returns, window)
    n = length(returns);
    rolling_sharpe = NaN(n, 1);
    
    for i = window:n
        window_returns = returns(i-window+1:i);
        window_returns = window_returns(~isnan(window_returns));
        
        if length(window_returns) > 10
            mean_return = mean(window_returns) * 252;
            vol_return = std(window_returns) * sqrt(252);
            if vol_return > 0
                rolling_sharpe(i) = mean_return / vol_return;
            end
        end
    end
end

function correlation_coefficient = correlation_coefficient(x, y)
    valid_idx = ~isnan(x) & ~isnan(y);
    x = x(valid_idx);
    y = y(valid_idx);
    
    if length(x) < 2
        correlation_coefficient = NaN;
        return;
    end
    
    x_centered = x - mean(x);
    y_centered = y - mean(y);
    
    numerator = sum(x_centered .* y_centered);
    denominator = sqrt(sum(x_centered.^2) * sum(y_centered.^2));
    
    if denominator == 0
        correlation_coefficient = NaN;
    else
        correlation_coefficient = numerator / denominator;
    end
end