% File location: OctaveMasterPro/utils/parallel_monte_carlo.m
% Parallel processing helper (from parallel_wrappers.m). Kept in its own
% file so it can be called from other scripts: Octave subfunctions in
% parallel_wrappers.m are only visible inside that file.

function results = parallel_monte_carlo(simulation_func, n_simulations, chunk_size)
    % Run Monte Carlo simulations in parallel
    % Usage: results = parallel_monte_carlo(@my_simulation, 10000, 1000)
    
    if nargin < 3
        chunk_size = ceil(n_simulations / get_optimal_workers());
    end
    
    fprintf('Running %d Monte Carlo simulations in parallel...\n', n_simulations);
    
    % Split simulations into chunks
    n_chunks = ceil(n_simulations / chunk_size);
    chunk_results = cell(n_chunks, 1);
    
    tic;
    if check_parallel_capability()
        % Parallel execution of chunks
        for chunk = 1:n_chunks
            chunk_start = (chunk - 1) * chunk_size + 1;
            chunk_end = min(chunk * chunk_size, n_simulations);
            current_chunk_size = chunk_end - chunk_start + 1;
            
            % Run simulations for this chunk
            chunk_data = zeros(1, current_chunk_size);
            for sim = 1:current_chunk_size
                chunk_data(sim) = simulation_func();
            end
            chunk_results{chunk} = chunk_data;
        end
        
    else
        % Serial execution
        all_results = zeros(1, n_simulations);
        for sim = 1:n_simulations
            all_results(sim) = simulation_func();
        end
        chunk_results{1} = all_results;
    end
    
    simulation_time = toc;
    
    % Combine results
    results = [];
    for chunk = 1:length(chunk_results)
        results = [results, chunk_results{chunk}];
    end
    
    fprintf('Monte Carlo completed: %d simulations in %.2f seconds\n', ...
            length(results), simulation_time);
    
    % Compute statistics
    result_stats = struct();
    result_stats.data = results;
    result_stats.mean = mean(results);
    result_stats.std = std(results);
    result_stats.min = min(results);
    result_stats.max = max(results);
    result_stats.n_simulations = n_simulations;
    result_stats.execution_time = simulation_time;
    
    results = result_stats;
end
