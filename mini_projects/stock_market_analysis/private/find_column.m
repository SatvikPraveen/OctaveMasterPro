% Location: mini_projects/stock_market_analysis/private/find_column.m
% (split out of data_loader.m so it can be called from other files)

function col_idx = find_column(headers, keywords)
    col_idx = 0;
    for i = 1:length(headers)
        header_lower = lower(headers{i});
        for j = 1:length(keywords)
            if contains(header_lower, keywords{j})
                col_idx = i;
                return;
            end
        end
    end
end
