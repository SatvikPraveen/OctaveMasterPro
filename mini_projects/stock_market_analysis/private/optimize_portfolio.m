% Location: mini_projects/stock_market_analysis/private/optimize_portfolio.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function weights = optimize_portfolio(returns_matrix, varargin)
    % Modern Portfolio Theory optimization
    
    method = 'max_sharpe';
    target_return = 0.1;
    risk_free_rate = 0.02;
    constraints = 'long_only';
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'method', method = varargin{i+1};
            case 'target_return', target_return = varargin{i+1};
            case 'risk_free_rate', risk_free_rate = varargin{i+1};
            case 'constraints', constraints = varargin{i+1};
        end
    end
    
    mean_returns = mean(returns_matrix, 1)';
    cov_matrix = cov(returns_matrix);
    
    switch lower(method)
        case 'min_variance'
            weights = minimize_variance(cov_matrix, constraints);
        case 'max_sharpe'
            weights = maximize_sharpe(mean_returns, cov_matrix, risk_free_rate, constraints);
        case 'target_return'
            weights = target_return_optimization(mean_returns, cov_matrix, target_return, constraints);
    end
    
    portfolio_return = weights' * mean_returns * 252;
    portfolio_vol = sqrt(weights' * cov_matrix * weights) * sqrt(252);
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol;
    
    fprintf('\nOptimal Portfolio (%s):\n', method);
    fprintf('Expected Annual Return: %.2f%%\n', portfolio_return * 100);
    fprintf('Annual Volatility: %.2f%%\n', portfolio_vol * 100);
    fprintf('Sharpe Ratio: %.3f\n', sharpe);
    fprintf('Weights: ');
    fprintf('%.1f%% ', weights * 100);
    fprintf('\n');
end
