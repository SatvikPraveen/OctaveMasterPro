% Location: mini_projects/stock_market_analysis/private/demo_data_loading.m
% (split out of data_loader.m so it can be called from other files)

function demo_data_loading()
    % Demonstrate data loading capabilities
    
    fprintf('\n--- Data Loading Demonstration ---\n');
    
    % Generate multiple synthetic stocks
    stocks = {'AAPL', 'GOOGL', 'MSFT', 'TSLA'};
    stock_data = cell(length(stocks), 1);
    
    rand('state', 2024);  % reproducible stock parameters
    params = rand(length(stocks), 3);
    for i = 1:length(stocks)
        fprintf('Generating data for %s...\n', stocks{i});
        stock_data{i} = generate_synthetic_stock_data('days', 252, 'seed', 100 + i, ...
            'initial_price', 50 + 100*params(i, 1), ...
            'return', 0.05 + 0.15*params(i, 2), ...
            'volatility', 0.15 + 0.25*params(i, 3));
        stock_data{i} = add_derived_fields(stock_data{i});  % adds .returns
    end
    
    % Plot all stocks
    figure('Position', [100, 100, 1400, 800]);
    
    subplot(2, 2, 1);
    colors = {'b', 'r', 'g', 'm'};
    for i = 1:length(stocks)
        plot(stock_data{i}.close, colors{i}, 'LineWidth', 1.5); hold on;
    end
    legend(stocks, 'Location', 'best');
    title('Stock Price Comparison');
    xlabel('Trading Days'); ylabel('Price ($)'); grid on;
    
    subplot(2, 2, 2);
    for i = 1:length(stocks)
        returns = stock_data{i}.returns(2:end) * 100;
        plot(returns, colors{i}, 'LineWidth', 1); hold on;
    end
    legend(stocks, 'Location', 'best');
    title('Daily Returns Comparison');
    xlabel('Trading Days'); ylabel('Return (%)'); grid on;
    
    subplot(2, 2, 3);
    volumes = zeros(length(stocks), 1);
    volatilities = zeros(length(stocks), 1);
    for i = 1:length(stocks)
        volumes(i) = mean(stock_data{i}.volume);
        volatilities(i) = std(stock_data{i}.returns(2:end)) * sqrt(252) * 100;
    end
    
    [ax, h1, h2] = plotyy(1:length(stocks), volumes/1e6, 1:length(stocks), volatilities);
    set(h1, 'LineStyle', '-', 'Marker', 'o', 'LineWidth', 2, 'MarkerSize', 8);
    set(h2, 'LineStyle', '-', 'Marker', 's', 'LineWidth', 2, 'MarkerSize', 8);
    set(ax(1), 'XTick', 1:length(stocks), 'XTickLabel', stocks);
    set(ax(2), 'XTick', 1:length(stocks), 'XTickLabel', stocks);
    ylabel(ax(1), 'Avg Volume (M)');
    ylabel(ax(2), 'Volatility (%)');
    title('Volume vs Volatility');
    grid on;
    
    subplot(2, 2, 4);
    total_returns = zeros(length(stocks), 1);
    for i = 1:length(stocks)
        total_returns(i) = (stock_data{i}.close(end) / stock_data{i}.close(1) - 1) * 100;
    end
    
    bar(total_returns);
    set(gca, 'XTickLabel', stocks);
    title('Total Returns');
    ylabel('Return (%)'); grid on;
    
    sgtitle('Stock Data Loading Demonstration');
    
    fprintf('Data loading demonstration complete.\n');
end
