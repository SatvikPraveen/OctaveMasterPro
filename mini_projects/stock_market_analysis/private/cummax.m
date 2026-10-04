% Location: mini_projects/stock_market_analysis/private/cummax.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function running_max = cummax(values)
    running_max = NaN(size(values));
    running_max(1) = values(1);
    
    for i = 2:length(values)
        running_max(i) = max(running_max(i-1), values(i));
    end
end
