% Location: mini_projects/stock_market_analysis/technical_indicators.m
% Technical Indicators for Stock Market Analysis

function sma = simple_moving_average(prices, window)
    % Simple Moving Average
    
    n = length(prices);
    sma = NaN(n, 1);
    
    for i = window:n
        sma(i) = mean(prices(i-window+1:i));
    end
end

function ema = exponential_moving_average(prices, window)
    % Exponential Moving Average
    
    n = length(prices);
    ema = NaN(n, 1);
    alpha = 2 / (window + 1);
    
    % Initialize with SMA
    ema(window) = mean(prices(1:window));
    
    for i = window+1:n
        ema(i) = alpha * prices(i) + (1 - alpha) * ema(i-1);
    end
end

function rsi = relative_strength_index(prices, period)
    % Relative Strength Index
    
    n = length(prices);
    rsi = NaN(n, 1);
    
    % Calculate price changes
    price_changes = [NaN; diff(prices)];
    
    % Separate gains and losses
    gains = max(price_changes, 0);
    losses = -min(price_changes, 0);
    
    for i = period+1:n
        avg_gain = mean(gains(i-period+1:i));
        avg_loss = mean(losses(i-period+1:i));
        
        if avg_loss == 0
            rsi(i) = 100;
        else
            rs = avg_gain / avg_loss;
            rsi(i) = 100 - (100 / (1 + rs));
        end
    end
end

function [macd_line, signal_line, histogram] = macd_indicator(prices, varargin)
    % MACD (Moving Average Convergence Divergence)
    
    % Default parameters
    fast_period = 12;
    slow_period = 26;
    signal_period = 9;
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'fast', fast_period = varargin{i+1};
            case 'slow', slow_period = varargin{i+1};
            case 'signal', signal_period = varargin{i+1};
        end
    end
    
    % Calculate EMAs
    ema_fast = exponential_moving_average(prices, fast_period);
    ema_slow = exponential_moving_average(prices, slow_period);
    
    % MACD line
    macd_line = ema_fast - ema_slow;
    
    % Signal line (EMA of MACD)
    valid_macd = macd_line(~isnan(macd_line));
    if length(valid_macd) >= signal_period
        signal_ema = exponential_moving_average(valid_macd, signal_period);
        signal_line = NaN(size(macd_line));
        signal_line(~isnan(macd_line)) = signal_ema;
    else
        signal_line = NaN(size(macd_line));
    end
    
    % Histogram
    histogram = macd_line - signal_line;
end

function [upper_band, middle_band, lower_band] = bollinger_bands(prices, period, num_std)
    % Bollinger Bands
    
    if nargin < 3, num_std = 2; end
    
    n = length(prices);
    upper_band = NaN(n, 1);
    middle_band = NaN(n, 1);
    lower_band = NaN(n, 1);
    
    for i = period:n
        window = prices(i-period+1:i);
        sma_val = mean(window);
        std_val = std(window);
        
        middle_band(i) = sma_val;
        upper_band(i) = sma_val + num_std * std_val;
        lower_band(i) = sma_val - num_std * std_val;
    end
end

function stoch = stochastic_oscillator(high, low, close, k_period, d_period)
    % Stochastic Oscillator
    
    n = length(close);
    percent_k = NaN(n, 1);
    
    for i = k_period:n
        highest_high = max(high(i-k_period+1:i));
        lowest_low = min(low(i-k_period+1:i));
        
        if highest_high > lowest_low
            percent_k(i) = 100 * (close(i) - lowest_low) / (highest_high - lowest_low);
        else
            percent_k(i) = 50; % Neutral value
        end
    end
    
    % %D is SMA of %K
    percent_d = simple_moving_average(percent_k, d_period);
    
    stoch.percent_k = percent_k;
    stoch.percent_d = percent_d;
end

function atr = average_true_range(high, low, close, period)
    % Average True Range
    
    n = length(close);
    
    % Calculate true range
    tr = NaN(n, 1);
    for i = 2:n
        tr1 = high(i) - low(i);
        tr2 = abs(high(i) - close(i-1));
        tr3 = abs(low(i) - close(i-1));
        tr(i) = max([tr1, tr2, tr3]);
    end
    
    % Average true range
    atr = simple_moving_average(tr, period);
end

function williams_r = williams_percent_r(high, low, close, period)
    % Williams %R Oscillator
    
    n = length(close);
    williams_r = NaN(n, 1);
    
    for i = period:n
        highest_high = max(high(i-period+1:i));
        lowest_low = min(low(i-period+1:i));
        
        if highest_high > lowest_low
            williams_r(i) = -100 * (highest_high - close(i)) / (highest_high - lowest_low);
        else
            williams_r(i) = -50;
        end
    end
end

function cci = commodity_channel_index(high, low, close, period)
    % Commodity Channel Index
    
    typical_price = (high + low + close) / 3;
    sma_tp = simple_moving_average(typical_price, period);
    
    n = length(close);
    cci = NaN(n, 1);
    
    for i = period:n
        mean_deviation = mean(abs(typical_price(i-period+1:i) - sma_tp(i)));
        if mean_deviation > 0
            cci(i) = (typical_price(i) - sma_tp(i)) / (0.015 * mean_deviation);
        end
    end
end

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