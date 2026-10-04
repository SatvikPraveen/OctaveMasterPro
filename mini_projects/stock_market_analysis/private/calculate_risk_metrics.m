% Location: mini_projects/stock_market_analysis/private/calculate_risk_metrics.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function risk_metrics = calculate_risk_metrics(portfolio_returns)
    clean_returns = portfolio_returns(~isnan(portfolio_returns));
    
    risk_metrics.volatility = std(clean_returns) * sqrt(252);
    
    downside_returns = min(clean_returns, 0);
    risk_metrics.downside_deviation = sqrt(mean(downside_returns.^2)) * sqrt(252);
    
    risk_metrics.var_95 = quantile_simple(clean_returns, 0.05);
    risk_metrics.var_99 = quantile_simple(clean_returns, 0.01);
    risk_metrics.cvar_95 = mean(clean_returns(clean_returns <= risk_metrics.var_95));
    
    cumulative_values = cumprod(1 + clean_returns);
    risk_metrics.max_drawdown = calculate_max_drawdown(cumulative_values);
    
    if risk_metrics.downside_deviation > 0
        risk_metrics.sortino_ratio = mean(clean_returns) * 252 / risk_metrics.downside_deviation;
    else
        risk_metrics.sortino_ratio = inf;
    end
    
    if risk_metrics.max_drawdown < 0
        risk_metrics.calmar_ratio = mean(clean_returns) * 252 / abs(risk_metrics.max_drawdown);
    else
        risk_metrics.calmar_ratio = inf;
    end
end
