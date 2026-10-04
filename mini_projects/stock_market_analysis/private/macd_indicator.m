% Location: mini_projects/stock_market_analysis/private/macd_indicator.m
% (split out of technical_indicators.m so it can be called from other files)

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
