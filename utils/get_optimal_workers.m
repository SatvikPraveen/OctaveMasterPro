% File location: OctaveMasterPro/utils/get_optimal_workers.m
% Parallel processing helper (from parallel_wrappers.m). Kept in its own
% file so it can be called from other scripts: Octave subfunctions in
% parallel_wrappers.m are only visible inside that file.

function optimal_workers = get_optimal_workers()
    % Determine optimal number of workers based on system resources
    % Usage: n_workers = get_optimal_workers()
    
    try
        % Try to get actual CPU count
        if exist('nproc', 'builtin')
            n_cores = nproc();
        else
            % Fallback method
            if isunix()
                [status, result] = system('nproc');
                if status == 0
                    n_cores = str2double(strtrim(result));
                else
                    n_cores = 4; % Conservative default
                end
            else
                n_cores = 4; % Windows/other default
            end
        end
        
        % Use cores - 1 to leave one for system
        optimal_workers = max(1, n_cores - 1);
        
        % Check memory constraints
        try
            memory_info = memory();
            available_memory_gb = memory_info.MemAvailableAllArrays / 1e9;
            
            % Limit workers based on memory (assume 0.5GB per worker minimum)
            memory_limited_workers = floor(available_memory_gb / 0.5);
            optimal_workers = min(optimal_workers, memory_limited_workers);
        catch
            % If memory info unavailable, be conservative
            optimal_workers = min(optimal_workers, 4);
        end
        
    catch
        optimal_workers = 2; % Very conservative fallback
    end
    
    optimal_workers = max(1, optimal_workers); % At least 1 worker
    fprintf('Optimal workers for this system: %d\n', optimal_workers);
end
