% Location: mini_projects/stock_market_analysis/private/generate_date_vector.m
% (split out of data_loader.m so it can be called from other files)

function dates = generate_date_vector(num_days)
    start_date = datenum('2020-01-01');
    dates = start_date + (0:num_days-1);
end
