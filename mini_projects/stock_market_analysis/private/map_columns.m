% Location: mini_projects/stock_market_analysis/private/map_columns.m
% (split out of data_loader.m so it can be called from other files)

function col_map = map_columns(headers)
    % Map CSV columns to standard field names
    
    col_map.open = find_column(headers, {'open'});
    col_map.high = find_column(headers, {'high'});
    col_map.low = find_column(headers, {'low'});
    col_map.close = find_column(headers, {'close'});
    col_map.volume = find_column(headers, {'volume', 'vol'});
    col_map.adj_close = find_column(headers, {'adj', 'adjusted'});
    
    if col_map.open == 0, col_map.open = 2; end
    if col_map.high == 0, col_map.high = 3; end
    if col_map.low == 0, col_map.low = 4; end
    if col_map.close == 0, col_map.close = 5; end
    if col_map.volume == 0, col_map.volume = 6; end
end
