% Location: mini_projects/stock_market_analysis/private/target_return_optimization.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function weights = target_return_optimization(mean_returns, cov_matrix, target_return, constraints)
    num_assets = length(mean_returns);
    target_daily = target_return / 252;
    
    ones_vec = ones(num_assets, 1);
    inv_cov = inv(cov_matrix + 1e-8 * eye(num_assets));
    
    A = [mean_returns'; ones_vec'];
    b = [target_daily; 1];
    
    lambda = inv(A * inv_cov * A') * (b - A * inv_cov * ones_vec);
    weights = inv_cov * (ones_vec + A' * lambda);
    
    if strcmp(constraints, 'long_only')
        weights = max(weights, 0);
        weights = weights / sum(weights);
    end
end
