% Location: mini_projects/stock_market_analysis/private/calculate_efficient_frontier.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function efficient_frontier = calculate_efficient_frontier(returns_matrix, num_points)
    if nargin < 2, num_points = 50; end
    
    mean_returns = mean(returns_matrix, 1)';
    cov_matrix = cov(returns_matrix);
    
    min_return = min(mean_returns) * 252;
    max_return = max(mean_returns) * 252;
    target_returns = linspace(min_return, max_return, num_points);
    
    efficient_frontier.returns = NaN(num_points, 1);
    efficient_frontier.volatilities = NaN(num_points, 1);
    efficient_frontier.sharpe_ratios = NaN(num_points, 1);
    efficient_frontier.weights = NaN(num_points, length(mean_returns));
    
    risk_free_rate = 0.02;
    
    for i = 1:num_points
        try
            weights = target_return_optimization(mean_returns, cov_matrix, target_returns(i), 'long_only');
            
            port_return = weights' * mean_returns * 252;
            port_vol = sqrt(weights' * cov_matrix * weights) * sqrt(252);
            sharpe = (port_return - risk_free_rate) / port_vol;
            
            efficient_frontier.returns(i) = port_return;
            efficient_frontier.volatilities(i) = port_vol;
            efficient_frontier.sharpe_ratios(i) = sharpe;
            efficient_frontier.weights(i, :) = weights';
        catch
            % Skip problematic points
            continue;
        end
    end
end
