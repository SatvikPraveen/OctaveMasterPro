% Location: mini_projects/stock_market_analysis/private/generate_correlated_returns.m
% (split out of portfolio_optimizer.m so it can be called from other files)

function returns_matrix = generate_correlated_returns(num_assets, num_days)
    % Generate correlated asset returns for demonstration
    
    # Base parameters
    annual_returns = [0.08, 0.06, 0.10, 0.07]; % Different expected returns
    annual_vols = [0.20, 0.15, 0.25, 0.18]; % Different volatilities
    
    # Ensure we have enough parameters
    if num_assets > length(annual_returns)
        annual_returns = [annual_returns, repmat(0.08, 1, num_assets - length(annual_returns))];
        annual_vols = [annual_vols, repmat(0.20, 1, num_assets - length(annual_vols))];
    end
    
    # Convert to daily
    daily_returns = annual_returns(1:num_assets) / 252;
    daily_vols = annual_vols(1:num_assets) / sqrt(252);
    
    # Create correlation structure
    correlation_matrix = eye(num_assets);
    for i = 1:num_assets
        for j = i+1:num_assets
            correlation_matrix(i,j) = 0.3 + 0.4 * rand(); % Random correlations 0.3-0.7
            correlation_matrix(j,i) = correlation_matrix(i,j);
        end
    end
    
    # Generate correlated returns using Cholesky decomposition
    L = chol(correlation_matrix, 'lower');
    
    rand('state', 456); randn('state', 456);
    independent_returns = randn(num_days, num_assets);
    correlated_returns = independent_returns * L';
    
    # Scale to desired means and volatilities
    returns_matrix = NaN(num_days, num_assets);
    for i = 1:num_assets
        returns_matrix(:, i) = daily_returns(i) + daily_vols(i) * correlated_returns(:, i);
    end
end
