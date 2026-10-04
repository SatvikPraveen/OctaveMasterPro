% Location: mini_projects/stock_market_analysis/private/analyze_returns.m
% (split out of price_analysis.m so it can be called from other files)

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
