% Location: mini_projects/stock_market_analysis/private/demo_technical_indicators.m
% (split out of technical_indicators.m so it can be called from other files)

function demo_technical_indicators()
    % Demonstrate all technical indicators
    
    fprintf('\n--- Technical Indicators Demonstration ---\n');
    
    % Load sample data
    data = load_stock_data('');
    prices = data.close;
    high = data.high;
    low = data.low;
    volume = data.volume;
    
    % Calculate indicators
    fprintf('Calculating technical indicators...\n');
    
    sma_20 = simple_moving_average(prices, 20);
    sma_50 = simple_moving_average(prices, 50);
    ema_12 = exponential_moving_average(prices, 12);
    ema_26 = exponential_moving_average(prices, 26);
    
    rsi_14 = relative_strength_index(prices, 14);
    [macd, signal, macd_hist] = macd_indicator(prices);
    [bb_upper, bb_middle, bb_lower] = bollinger_bands(prices, 20, 2);
    
    stoch = stochastic_oscillator(high, low, prices, 14, 3);
    atr = average_true_range(high, low, prices, 14);
    williams_r = williams_percent_r(high, low, prices, 14);
    
    % Plot indicators
    figure('Position', [50, 50, 1400, 1200]);
    
    % Price and moving averages
    subplot(4, 2, 1);
    plot(prices, 'k', 'LineWidth', 1.5); hold on;
    plot(sma_20, 'b', 'LineWidth', 1);
    plot(sma_50, 'r', 'LineWidth', 1);
    plot(ema_12, 'g', 'LineWidth', 1);
    legend('Price', 'SMA-20', 'SMA-50', 'EMA-12', 'Location', 'best');
    title('Price and Moving Averages');
    ylabel('Price ($)'); grid on;
    
    # Bollinger Bands
    subplot(4, 2, 2);
    plot(prices, 'k', 'LineWidth', 1.5); hold on;
    plot(bb_upper, 'r--', 'LineWidth', 1);
    plot(bb_middle, 'b', 'LineWidth', 1);
    plot(bb_lower, 'r--', 'LineWidth', 1);
    fill([1:length(prices), length(prices):-1:1], [bb_upper', bb_lower(end:-1:1)'], 'r', 'FaceAlpha', 0.1);
    legend('Price', 'Upper Band', 'Middle Band', 'Lower Band', 'Location', 'best');
    title('Bollinger Bands');
    ylabel('Price ($)'); grid on;
    
    % RSI
    subplot(4, 2, 3);
    plot(rsi_14, 'purple', 'LineWidth', 1.5);
    hold on; plot([1, length(rsi_14)], [70, 70], 'r--'); plot([1, length(rsi_14)], [30, 30], 'g--');
    title('RSI (14-period)');
    ylabel('RSI'); ylim([0, 100]); grid on;
    
    % MACD
    subplot(4, 2, 4);
    plot(macd, 'b', 'LineWidth', 1.5); hold on;
    plot(signal, 'r', 'LineWidth', 1);
    bar(macd_hist, 'FaceColor', 'g', 'FaceAlpha', 0.5);
    legend('MACD', 'Signal', 'Histogram', 'Location', 'best');
    title('MACD Indicator');
    ylabel('MACD'); grid on;
    
    % Stochastic
    subplot(4, 2, 5);
    plot(stoch.percent_k, 'b', 'LineWidth', 1.5); hold on;
    plot(stoch.percent_d, 'r', 'LineWidth', 1);
    plot([1, length(stoch.percent_k)], [80, 80], 'r--');
    plot([1, length(stoch.percent_k)], [20, 20], 'g--');
    legend('%K', '%D', 'Location', 'best');
    title('Stochastic Oscillator');
    ylabel('Stochastic'); ylim([0, 100]); grid on;
    
    % Williams %R
    subplot(4, 2, 6);
    plot(williams_r, 'magenta', 'LineWidth', 1.5);
    hold on; plot([1, length(williams_r)], [-20, -20], 'r--'); plot([1, length(williams_r)], [-80, -80], 'g--');
    title('Williams %R');
    ylabel('Williams %R'); ylim([-100, 0]); grid on;
    
    % Volume analysis
    subplot(4, 2, 7);
    bar(volume/1e6, 'FaceColor', 'cyan', 'FaceAlpha', 0.7);
    title('Volume');
    ylabel('Volume (M)'); grid on;
    
    # Average True Range
    subplot(4, 2, 8);
    plot(atr, 'orange', 'LineWidth', 1.5);
    title('Average True Range (14)');
    ylabel('ATR'); grid on;
    
    sgtitle('Technical Indicators Dashboard');
    
    fprintf('Technical indicators calculation complete.\n');
    
    % Print latest values
    fprintf('\nLatest indicator values:\n');
    fprintf('SMA-20: $%.2f\n', sma_20(end));
    fprintf('RSI-14: %.1f\n', rsi_14(end));
    fprintf('MACD: %.3f\n', macd(end));
    fprintf('Williams %%R: %.1f\n', williams_r(end));
end
