% Location: mini_projects/stock_market_analysis/private/calculate_max_drawdown.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function max_dd = calculate_max_drawdown(cumulative_values)
    running_max = cummax(cumulative_values);
    drawdown = (cumulative_values - running_max) ./ running_max;
    max_dd = min(drawdown);
end
