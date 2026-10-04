% Location: mini_projects/stock_market_analysis/private/minimize_variance.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function weights = minimize_variance(cov_matrix, constraints)
    num_assets = size(cov_matrix, 1);
    ones_vec = ones(num_assets, 1);
    inv_cov = inv(cov_matrix + 1e-8 * eye(num_assets));
    
    weights = inv_cov * ones_vec / (ones_vec' * inv_cov * ones_vec);
    
    if strcmp(constraints, 'long_only')
        weights = max(weights, 0);
        weights = weights / sum(weights);
    end
end
