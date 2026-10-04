% Location: mini_projects/stock_market_analysis/private/maximize_sharpe.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function weights = maximize_sharpe(mean_returns, cov_matrix, risk_free_rate, constraints)
    num_assets = length(mean_returns);
    excess_returns = mean_returns - risk_free_rate / 252;
    
    inv_cov = inv(cov_matrix + 1e-8 * eye(num_assets));
    weights = inv_cov * excess_returns;
    weights = weights / sum(weights);
    
    if strcmp(constraints, 'long_only')
        weights = max(weights, 0);
        if sum(weights) > 0
            weights = weights / sum(weights);
        else
            weights = ones(num_assets, 1) / num_assets;
        end
    end
end
