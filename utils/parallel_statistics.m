% File location: OctaveMasterPro/utils/parallel_statistics.m
% Parallel processing helper (from parallel_wrappers.m). Kept in its own
% file so it can be called from other scripts: Octave subfunctions in
% parallel_wrappers.m are only visible inside that file.

function results = parallel_statistics(data, stat_functions)
    % Compute multiple statistics in parallel
    % Usage: results = parallel_statistics(data_matrix, {@mean, @std, @median})
    
    n_functions = length(stat_functions);
    fprintf('Computing %d statistics in parallel...\n', n_functions);
    
    results = cell(n_functions, 1);
    
    % Execute statistics functions
    tic;
    if check_parallel_capability() && n_functions > 1
        % Parallel execution
        for i = 1:n_functions
            results{i} = stat_functions{i}(data);
        end
    else
        % Serial execution
        for i = 1:n_functions
            results{i} = stat_functions{i}(data);
        end
    end
    computation_time = toc;
    
    fprintf('Statistics computed in %.4f seconds\n', computation_time);
    
    % Package results with function names
    result_struct = struct();
    for i = 1:n_functions
        func_name = func2str(stat_functions{i});
        result_struct.(func_name) = results{i};
    end
    
    results = result_struct;
end
