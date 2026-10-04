% Location: mini_projects/stock_market_analysis/private/validate_stock_data.m
% (split out of data_loader.m so it can be called from other files)

function validated_data = validate_stock_data(data)
    validated_data = data;
    issues_found = 0;
    
    if any(data.open <= 0) || any(data.high <= 0) || any(data.low <= 0) || any(data.close <= 0)
        valid_idx = data.open > 0 & data.high > 0 & data.low > 0 & data.close > 0;
        validated_data = filter_data_by_index(data, valid_idx);
        issues_found = issues_found + sum(~valid_idx);
    end
    
    ohlc_valid = validated_data.high >= validated_data.low & ...
                 validated_data.high >= validated_data.open & ...
                 validated_data.high >= validated_data.close & ...
                 validated_data.low <= validated_data.open & ...
                 validated_data.low <= validated_data.close;
    
    if ~all(ohlc_valid)
        issues_found = issues_found + sum(~ohlc_valid);
        for i = 1:length(validated_data.close)
            if ~ohlc_valid(i)
                prices = [validated_data.open(i), validated_data.close(i)];
                validated_data.high(i) = max([prices, validated_data.high(i)]);
                validated_data.low(i) = min([prices, validated_data.low(i)]);
            end
        end
    end
    
    if issues_found == 0
        fprintf('Data validation passed.\n');
    else
        fprintf('Data validation completed. %d issues corrected.\n', issues_found);
    end
end
