% Location: mini_projects/stock_market_analysis/private/fill_missing_values.m
% (split out of data_loader.m so it can be called from other files)

function filled_data = fill_missing_values(data, method)
    filled_data = data;
    
    fields_to_fill = {'open', 'high', 'low', 'close', 'adj_close', 'volume'};
    
    for f = 1:length(fields_to_fill)
        field_name = fields_to_fill{f};
        if isfield(data, field_name)
            values = data.(field_name);
            
            switch lower(method)
                case 'forward'
                    % Forward fill
                    for i = 2:length(values)
                        if isnan(values(i)) || values(i) == 0
                            values(i) = values(i-1);
                        end
                    end
                    
                case 'backward'
                    % Backward fill
                    for i = length(values)-1:-1:1
                        if isnan(values(i)) || values(i) == 0
                            values(i) = values(i+1);
                        end
                    end
                    
                case 'linear'
                    % Linear interpolation
                    valid_idx = ~isnan(values) & values > 0;
                    if sum(valid_idx) > 1
                        values = interp1(find(valid_idx), values(valid_idx), 1:length(values), 'linear', 'extrap');
                    end
            end
            
            filled_data.(field_name) = values;
        end
    end
end
