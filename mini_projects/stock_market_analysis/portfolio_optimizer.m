% Location: mini_projects/stock_market_analysis/portfolio_optimizer.m
% Portfolio Optimization and Risk Management

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

function risk_metrics = calculate_risk_metrics(portfolio_returns)
    % Calculate comprehensive risk metrics
    
    clean_returns = portfolio_returns(~isnan(portfolio_returns));
    
    risk_metrics.volatility = std(clean_returns) * sqrt(252);
    risk_metrics.downside_deviation = sqrt(mean(min(clean_returns, 0).^2)) * sqrt(252);
    risk_metrics.var_95 = quantile_simple(clean_returns, 0.05);
    risk_metrics.var_99 = quantile_simple(clean_returns, 0.01);
    risk_metrics.cvar_95 = mean(clean_returns(clean_returns <= risk_metrics.var_95));
    risk_metrics.max_drawdown = calculate_max_drawdown(cumprod(1 + clean_returns));
    risk_metrics.sortino_ratio = mean(clean_returns) * 252 / risk_metrics.downside_deviation;
    risk_metrics.calmar_ratio = mean(clean_returns) * 252 / abs(risk_metrics.max_drawdown);
end

function q = quantile_simple(data, p)
    sorted_data = sort(data);
    n = length(sorted_data);
    index = p * (n - 1) + 1;
    
    if index == round(index)
        q = sorted_data(round(index));
    else
        lower = floor(index);
        upper = ceil(index);
        weight = index - lower;
        q = sorted_data(lower) * (1 - weight) + sorted_data(upper) * weight;
    end
end

function max_dd = calculate_max_drawdown(cumulative_values)
    running_max = cummax(cumulative_values);
    drawdown = (cumulative_values - running_max) ./ running_max;
    max_dd = min(drawdown);
end

function running_max = cummax(values)
    running_max = NaN(size(values));
    running_max(1) = values(1);
    
    for i = 2:length(values)
        running_max(i) = max(running_max(i-1), values(i));
    end
end

function correlation_matrix = calculate_correlation_matrix(returns_matrix)
    % Calculate correlation matrix with visualization
    
    correlation_matrix = corr(returns_matrix);
    
    figure('Position', [200, 200, 800, 600]);
    
    imagesc(correlation_matrix);
    colorbar;
    colormap('cool');
    
    num_assets = size(returns_matrix, 2);
    asset_names = cell(num_assets, 1);
    for i = 1:num_assets
        asset_names{i} = sprintf('Asset %d', i);
    end
    
    set(gca, 'XTick', 1:num_assets, 'XTickLabel', asset_names);
    set(gca, 'YTick', 1:num_assets, 'YTickLabel', asset_names);
    title('Asset Correlation Matrix');
    
    % Add correlation values as text
    for i = 1:num_assets
        for j = 1:num_assets
            text(j, i, sprintf('%.2f', correlation_matrix(i,j)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
        end
    end
end

function backtest_results = backtest_portfolio(returns_matrix, weights, varargin)
    % Backtest portfolio performance
    
    rebalance_frequency = 'monthly'; % 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_cost = 0.001; % 0.1% per transaction
    
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'rebalance', rebalance_frequency = varargin{i+1};
            case 'cost', transaction_cost = varargin{i+1};
        end
    end
    
    [num_days, num_assets] = size(returns_matrix);
    
    % Determine rebalancing frequency
    switch lower(rebalance_frequency)
        case 'daily', rebal_freq = 1;
        case 'weekly', rebal_freq = 5;
        case 'monthly', rebal_freq = 21;
        case 'quarterly', rebal_freq = 63;
        otherwise, rebal_freq = 21;
    end
    
    # Initialize portfolio
    portfolio_value = 1;
    portfolio_weights = weights;
    portfolio_values = NaN(num_days, 1);
    actual_weights = NaN(num_days, num_assets);
    transaction_costs = 0;
    
    for day = 1:num_days
        if day == 1
            portfolio_values(day) = portfolio_value;
            actual_weights(day, :) = portfolio_weights';
        else
            # Calculate return
            daily_returns = returns_matrix(day, :);
            portfolio_return = portfolio_weights' * daily_returns';
            portfolio_value = portfolio_value * (1 + portfolio_return);
            
            # Update weights based on price movements
            new_weights = portfolio_weights .* (1 + daily_returns');
            new_weights = new_weights / sum(new_weights);
            
            portfolio_values(day) = portfolio_value;
            actual_weights(day, :) = new_weights';
            
            # Rebalance if needed
            if mod(day, rebal_freq) == 0
                weight_changes = abs(new_weights - weights);
                total_change = sum(weight_changes);
                transaction_costs = transaction_costs + total_change * transaction_cost * portfolio_value;
                portfolio_weights = weights;
            else
                portfolio_weights = new_weights;
            end
        end
    end
    
    # Calculate performance metrics
    total_return = (portfolio_values(end) - 1) * 100;
    annual_return = (portfolio_values(end)^(252/num_days) - 1) * 100;
    
    daily_portfolio_returns = [NaN; diff(log(portfolio_values))];
    annual_volatility = std(daily_portfolio_returns(~isnan(daily_portfolio_returns))) * sqrt(252) * 100;
    
    backtest_results.portfolio_values = portfolio_values;
    backtest_results.actual_weights = actual_weights;
    backtest_results.total_return = total_return;
    backtest_results.annual_return = annual_return;
    backtest_results.annual_volatility = annual_volatility;
    backtest_results.sharpe_ratio = annual_return / annual_volatility;
    backtest_results.transaction_costs = transaction_costs;
    backtest_results.max_drawdown = calculate_max_drawdown(portfolio_values) * 100;
    
    fprintf('\nBacktest Results:\n');
    fprintf('Total Return: %.2f%%\n', total_return);
    fprintf('Annual Return: %.2f%%\n', annual_return);
    fprintf('Annual Volatility: %.2f%%\n', annual_volatility);
    fprintf('Sharpe Ratio: %.3f\n', backtest_results.sharpe_ratio);
    fprintf('Max Drawdown: %.2f%%\n', backtest_results.max_drawdown);
    fprintf('Transaction Costs: $%.2f\n', transaction_costs);
end

function demo_portfolio_optimization()
    % Demonstrate portfolio optimization
    
    fprintf('\n--- Portfolio Optimization Demonstration ---\n');
    
    # Generate sample data for multiple assets
    num_assets = 4;
    num_days = 252;
    
    fprintf('Generating sample data for %d assets...\n', num_assets);
    
    returns_matrix = generate_correlated_returns(num_assets, num_days);
    asset_names = {'Tech Stock', 'Financial', 'Energy', 'Healthcare'};
    
    # Calculate efficient frontier
    fprintf('Calculating efficient frontier...\n');
    ef = calculate_efficient_frontier(returns_matrix, 30);
    
    # Optimize for different objectives
    weights_min_var = optimize_portfolio(returns_matrix, 'method', 'min_variance');
    weights_max_sharpe = optimize_portfolio(returns_matrix, 'method', 'max_sharpe');
    weights_target = optimize_portfolio(returns_matrix, 'method', 'target_return', 'target_return', 0.12);
    
    # Visualize results
    figure('Position', [50, 50, 1400, 1000]);
    
    # Efficient frontier
    subplot(2, 3, 1);
    plot(ef.volatilities * 100, ef.returns * 100, 'b-', 'LineWidth', 2);
    hold on;
    
    # Plot individual assets
    individual_returns = mean(returns_matrix, 1) * 252 * 100;
    individual_vols = std(returns_matrix, 1) * sqrt(252) * 100;
    scatter(individual_vols, individual_returns, 100, 'r', 'filled');
    
    # Plot optimal portfolios
    mv_return = weights_min_var' * mean(returns_matrix, 1)' * 252 * 100;
    mv_vol = sqrt(weights_min_var' * cov(returns_matrix) * weights_min_var) * sqrt(252) * 100;
    scatter(mv_vol, mv_return, 150, 'g', 'filled');
    
    ms_return = weights_max_sharpe' * mean(returns_matrix, 1)' * 252 * 100;
    ms_vol = sqrt(weights_max_sharpe' * cov(returns_matrix) * weights_max_sharpe) * sqrt(252) * 100;
    scatter(ms_vol, ms_return, 150, 'm', 'filled');
    
    xlabel('Volatility (%)'); ylabel('Expected Return (%)');
    title('Efficient Frontier');
    legend('Efficient Frontier', 'Individual Assets', 'Min Variance', 'Max Sharpe', 'Location', 'best');
    grid on;
    
    # Portfolio weights comparison
    subplot(2, 3, 2);
    weights_matrix = [weights_min_var, weights_max_sharpe, weights_target];
    bar(weights_matrix * 100);
    set(gca, 'XTickLabel', asset_names);
    legend('Min Var', 'Max Sharpe', 'Target Return', 'Location', 'best');
    title('Portfolio Weights Comparison');
    ylabel('Weight (%)'); grid on;
    
    # Risk-return scatter
    subplot(2, 3, 3);
    scatter(ef.volatilities * 100, ef.sharpe_ratios, 50, ef.sharpe_ratios, 'filled');
    colorbar;
    xlabel('Volatility (%)'); ylabel('Sharpe Ratio');
    title('Risk vs Sharpe Ratio');
    grid on;
    
    # Asset correlation heatmap
    subplot(2, 3, 4);
    corr_matrix = corr(returns_matrix);
    imagesc(corr_matrix);
    colorbar; colormap('cool');
    set(gca, 'XTick', 1:num_assets, 'XTickLabel', asset_names);
    set(gca, 'YTick', 1:num_assets, 'YTickLabel', asset_names);
    title('Asset Correlations');
    
    for i = 1:num_assets
        for j = 1:num_assets
            text(j, i, sprintf('%.2f', corr_matrix(i,j)), 'HorizontalAlignment', 'center', 'Color', 'white');
        end
    end
    
    # Backtest performance
    subplot(2, 3, 5);
    backtest_max_sharpe = backtest_portfolio(returns_matrix, weights_max_sharpe, 'rebalance', 'monthly');
    backtest_equal_weight = backtest_portfolio(returns_matrix, ones(num_assets, 1)/num_assets, 'rebalance', 'monthly');
    
    plot(cumprod(1 + [NaN; diff(log(backtest_max_sharpe.portfolio_values))]) - 1, 'b', 'LineWidth', 2);
    hold on;
    plot(cumprod(1 + [NaN; diff(log(backtest_equal_weight.portfolio_values))]) - 1, 'r', 'LineWidth', 1.5);
    
    legend('Max Sharpe Portfolio', 'Equal Weight Portfolio', 'Location', 'best');
    title('Backtest Performance');
    ylabel('Cumulative Return'); grid on;
    
    # Risk metrics comparison
    subplot(2, 3, 6);
    risk_max_sharpe = calculate_risk_metrics([NaN; diff(log(backtest_max_sharpe.portfolio_values))]);
    risk_equal_weight = calculate_risk_metrics([NaN; diff(log(backtest_equal_weight.portfolio_values))]);
    
    metrics = {'Volatility', 'Sharpe', 'Max DD'};
    max_sharpe_vals = [risk_max_sharpe.volatility*100, risk_max_sharpe.sortino_ratio, abs(risk_max_sharpe.max_drawdown)*100];
    equal_weight_vals = [risk_equal_weight.volatility*100, risk_equal_weight.sortino_ratio, abs(risk_equal_weight.max_drawdown)*100];
    
    x = 1:length(metrics);
    bar(x-0.2, max_sharpe_vals, 0.4); hold on;
    bar(x+0.2, equal_weight_vals, 0.4);
    
    set(gca, 'XTickLabel', metrics);
    legend('Max Sharpe', 'Equal Weight', 'Location', 'best');
    title('Risk Metrics Comparison');
    ylabel('Values'); grid on;
    
    sgtitle('Portfolio Optimization Analysis');
    
    fprintf('Portfolio optimization demonstration complete.\n');
end

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