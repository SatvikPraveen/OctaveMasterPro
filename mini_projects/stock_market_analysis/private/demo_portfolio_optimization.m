% Location: mini_projects/stock_market_analysis/private/demo_portfolio_optimization.m
% (split out of portfolio_optimizer.m so it can be called from other files)

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
