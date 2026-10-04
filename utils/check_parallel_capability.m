% File location: OctaveMasterPro/utils/check_parallel_capability.m
% Parallel processing helper (from parallel_wrappers.m). Kept in its own
% file so it can be called from other scripts: Octave subfunctions in
% parallel_wrappers.m are only visible inside that file.

function is_available = check_parallel_capability()
    % Check if parallel processing is available and functional
    % Usage: is_available = check_parallel_capability()
    
    fprintf('Checking parallel processing capability...\n');
    
    % Check if parallel package is installed
    pkg_list = pkg('list');
    parallel_installed = false;
    
    for i = 1:length(pkg_list)
        if strcmp(pkg_list{i}.name, 'parallel')
            parallel_installed = true;
            if pkg_list{i}.loaded
                fprintf('  ✓ Parallel package loaded\n');
            else
                fprintf('  ! Parallel package installed but not loaded\n');
                pkg load parallel;
                fprintf('  ✓ Parallel package now loaded\n');
            end
            break;
        end
    end
    
    if ~parallel_installed
        fprintf('  ✗ Parallel package not installed\n');
        fprintf('  Install with: pkg install -forge parallel\n');
        is_available = false;
        return;
    end
    
    % Test basic parallel functionality
    try
        % Simple test with parfor (if available)
        test_data = 1:100;
        result = zeros(size(test_data));
        
        % Try parallel execution
        if exist('parfor', 'builtin')
            tic;
            parfor i = 1:length(test_data)
                result(i) = test_data(i)^2;
            end
            parallel_time = toc;
            
            fprintf('  ✓ parfor functionality working (%.4f seconds)\n', parallel_time);
        else
            fprintf('  ! parfor not available, using alternative parallel methods\n');
        end
        
        % Test system resources
        if exist('nproc', 'builtin')
            n_cores = nproc();
        else
            n_cores = 4; % Default assumption
        end
        
        fprintf('  ✓ Detected %d CPU cores\n', n_cores);
        
        is_available = true;
        
    catch ME
        fprintf('  ✗ Parallel functionality test failed: %s\n', ME.message);
        is_available = false;
    end
    
    % Memory check
    try
        memory_info = memory();
        available_memory = memory_info.MemAvailableAllArrays / 1e9;
        fprintf('  ✓ Available memory: %.1f GB\n', available_memory);
        
        if available_memory < 1
            fprintf('  ! Warning: Low memory may limit parallel processing\n');
        end
    catch
        fprintf('  ! Could not determine available memory\n');
    end
end
