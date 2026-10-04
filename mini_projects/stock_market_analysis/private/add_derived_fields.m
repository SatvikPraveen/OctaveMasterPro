% Location: mini_projects/stock_market_analysis/private/add_derived_fields.m
% (split out of data_loader.m so it can be called from other files)

function enhanced_data = add_derived_fields(data)
    % Add derived fields for analysis
    
    enhanced_data = data;
    
    % Daily returns
    enhanced_data.returns = [NaN; diff(log(data.close))];
    
    % Price changes
    enhanced_data.price_change = [NaN; diff(data.close)];
    enhanced_data.price_change_pct = enhanced_data.price_change ./ [NaN; data.close(1:end-1)] * 100;
    
    % Typical price
    enhanced_data.typical_price = (data.high + data.low + data.close) / 3;
    
    # True range for volatility calculation
    if length(data.close) > 1
        tr1 = data.high - data.low;
        tr2 = abs(data.high - [NaN; data.close(1:end-1)]);
        tr3 = abs(data.low - [NaN; data.close(1:end-1)]);
        enhanced_data.true_range = max([tr1, tr2, tr3], [], 2);
    else
        enhanced_data.true_range = data.high - data.low;
    end
    
    % Intraday range
    enhanced_data.intraday_range = (data.high - data.low) ./ data.close * 100;
    
    fprintf('Added derived fields: returns, price_change, typical_price, true_range.\n');
end
