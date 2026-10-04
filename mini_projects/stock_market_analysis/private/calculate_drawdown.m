% Location: mini_projects/stock_market_analysis/private/calculate_drawdown.m
% (split out of price_analysis.m so it can be called from other files)

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
