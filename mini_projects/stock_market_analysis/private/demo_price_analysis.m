% Location: mini_projects/stock_market_analysis/private/demo_price_analysis.m
% (split out of price_analysis.m so it can be called from other files)

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
    plot(rolling_vol, 'Color', [0.5 0 0.5], 'LineWidth', 1.5);
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
