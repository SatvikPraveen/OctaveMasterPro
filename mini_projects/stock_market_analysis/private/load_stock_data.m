% Location: mini_projects/stock_market_analysis/private/load_stock_data.m
% (split out of data_loader.m so it can be called from other files)

function data = load_stock_data(filename, varargin)
    % Load stock data from CSV file
    
    % Default parameters
    validate_data = true;
    fill_method = 'forward';
    
    % Parse arguments
    for i = 1:2:length(varargin)
        switch lower(varargin{i})
            case 'validate'
                validate_data = varargin{i+1};
            case 'fill_missing'
                fill_method = varargin{i+1};
        end
    end
    
    try
        if exist(filename, 'file')
            fprintf('Loading stock data from: %s\n', filename);
            
            fid = fopen(filename, 'r');
            header_line = fgetl(fid);
            fclose(fid);
            
            headers = strsplit(header_line, ',');
            headers = cellfun(@(x) strtrim(x), headers, 'UniformOutput', false);
            
            raw_data = csvread(filename, 1, 0);
            
            col_map = map_columns(headers);
            
            data.open = raw_data(:, col_map.open);
            data.high = raw_data(:, col_map.high);
            data.low = raw_data(:, col_map.low);
            data.close = raw_data(:, col_map.close);
            
            if col_map.volume > 0
                data.volume = raw_data(:, col_map.volume);
            else
                data.volume = ones(size(data.close));
            end
            
            if col_map.adj_close > 0
                data.adj_close = raw_data(:, col_map.adj_close);
            else
                data.adj_close = data.close;
            end
            
            data.dates = generate_date_vector(length(data.close));
            
        else
            fprintf('File not found. Generating synthetic stock data...\n');
            data = generate_synthetic_stock_data();
        end
        
        if validate_data
            data = validate_stock_data(data);
        end
        
        if ~strcmp(fill_method, 'none')
            data = fill_missing_values(data, fill_method);
        end
        
        data = add_derived_fields(data);
        
        fprintf('Stock data loaded: %d trading days.\n', length(data.close));
        
    catch err
        fprintf('Error loading: %s\n', err.message);
        data = add_derived_fields(generate_synthetic_stock_data());
    end
end
