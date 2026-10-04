% Location: mini_projects/stock_market_analysis/private/seasonal_analysis.m
% (split out of price_analysis.m so it can be called from other files)

function seasonal_analysis(prices, dates)
    % Analyze seasonal patterns in stock returns
    
    fprintf('\n--- Seasonal Analysis ---\n');
    
    # Extract month information (simplified)
    months = mod(floor(dates/30), 12) + 1; % Approximate month extraction
    
    monthly_returns = NaN(12, 1);
    monthly_volatility = NaN(12, 1);
    
    returns = [NaN; diff(log(prices))];
    
    for month = 1:12
        month_mask = months == month;
        if sum(month_mask) > 0
            month_returns = returns(month_mask);
            month_returns = month_returns(~isnan(month_returns));
            
            if length(month_returns) > 1
                monthly_returns(month) = mean(month_returns) * 252 * 100; % Annualized %
                monthly_volatility(month) = std(month_returns) * sqrt(252) * 100;
            end
        end
    end
    
    # Plot seasonal patterns
    figure('Position', [200, 200, 1200, 600]);
    
    subplot(1, 2, 1);
    bar(monthly_returns);
    month_names = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', ...
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
    set(gca, 'XTickLabel', month_names);
    title('Average Monthly Returns');
    ylabel('Annualized Return (%)'); grid on;
    
    subplot(1, 2, 2);
    bar(monthly_volatility);
    set(gca, 'XTickLabel', month_names);
    title('Average Monthly Volatility');
    ylabel('Annualized Volatility (%)'); grid on;
    
    sgtitle('Seasonal Analysis');
    
    fprintf('Seasonal analysis complete.\n');
end
