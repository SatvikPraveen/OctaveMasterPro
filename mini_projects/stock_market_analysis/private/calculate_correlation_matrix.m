% Location: mini_projects/stock_market_analysis/private/calculate_correlation_matrix.m
% (split out of portfolio_optimizer.m so it can be called from other files)

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
