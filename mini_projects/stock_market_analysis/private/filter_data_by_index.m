% Location: mini_projects/stock_market_analysis/private/filter_data_by_index.m
% (split out of data_loader.m so it can be called from other files)

function filtered_data = filter_data_by_index(data, valid_idx)
    fields = fieldnames(data);
    filtered_data = struct();
    
    for i = 1:length(fields)
        field_data = data.(fields{i});
        if length(field_data) == length(valid_idx)
            filtered_data.(fields{i}) = field_data(valid_idx);
        else
            filtered_data.(fields{i}) = field_data;
        end
    end
end
