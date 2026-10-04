% Location: mini_projects/stock_market_analysis/private/regime_analysis.m
% (split out of price_analysis.m so it can be called from other files)

function regime_analysis(prices, returns)
    % Market regime analysis (bull/bear markets)
    
    fprintf('\n--- Market Regime Analysis ---\n');
    
    # Simple regime detection based on moving averages
    sma_short = simple_moving_average(prices, 50);
    sma_long = simple_moving_average(prices, 200);
    
    # Bull market: short MA > long MA
    bull_market = sma_short > sma_long;
    
    # Find regime changes
    regime_changes = [false; diff(bull_market) ~= 0];
    change_indices = find(regime_changes);
    
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 1, 1);
    plot(prices, 'k', 'LineWidth', 1.5); hold on;
    plot(sma_short, 'b', 'LineWidth', 1);
    plot(sma_long, 'r', 'LineWidth', 1);
    
    # Color background by regime
    y_limits = ylim;
    for i = 1:length(change_indices)-1
        start_idx = change_indices(i);
        end_idx = change_indices(i+1) - 1;
        
        if bull_market(start_idx)
            fill([start_idx, end_idx, end_idx, start_idx], ...
                 [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
                 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        else
            fill([start_idx, end_idx, end_idx, start_idx], ...
                 [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
                 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        end
    end
    
    legend('Price', 'SMA-50', 'SMA-200', 'Location', 'best');
    title('Market Regimes (Green=Bull, Red=Bear)');
    ylabel('Price ($)'); grid on;
    
    subplot(2, 1, 2);
    cumulative_returns = cumprod(1 + returns(~isnan(returns))) - 1;
    plot(cumulative_returns * 100, 'k', 'LineWidth', 2);
    title('Cumulative Returns');
    xlabel('Trading Days'); ylabel('Cumulative Return (%)'); grid on;
    
    # Calculate regime statistics
    bull_periods = sum(bull_market(~isnan(bull_market)));
    total_periods = sum(~isnan(bull_market));
    bull_percentage = bull_periods / total_periods * 100;
    
    fprintf('Market was in bull regime %.1f%% of the time.\n', bull_percentage);
    fprintf('Number of regime changes: %d\n', length(change_indices));
end
