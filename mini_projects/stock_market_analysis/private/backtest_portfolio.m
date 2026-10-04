% Location: mini_projects/stock_market_analysis/private/backtest_portfolio.m
% (split out of portfolio_optimizer.m so it can be called from other files)

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
