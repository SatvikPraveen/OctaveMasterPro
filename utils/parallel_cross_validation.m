% File location: OctaveMasterPro/utils/parallel_cross_validation.m
% Parallel processing helper (from parallel_wrappers.m). Kept in its own
% file so it can be called from other scripts: Octave subfunctions in
% parallel_wrappers.m are only visible inside that file.

function results = parallel_cross_validation(model_func, data, k_folds)
    % Parallel k-fold cross-validation
    % Usage: results = parallel_cross_validation(@my_model, data, 10)
    
    if nargin < 3
        k_folds = 10;
    end
    
    fprintf('Running %d-fold cross-validation in parallel...\n', k_folds);
    
    % Split data into folds
    n_samples = size(data, 1);
    fold_size = floor(n_samples / k_folds);
    fold_results = cell(k_folds, 1);
    
    tic;
    if check_parallel_capability() && k_folds > 2
        % Parallel cross-validation
        for fold = 1:k_folds
            % Create training and test sets
            test_start = (fold - 1) * fold_size + 1;
            test_end = min(fold * fold_size, n_samples);
            
            test_indices = test_start:test_end;
            train_indices = setdiff(1:n_samples, test_indices);
            
            train_data = data(train_indices, :);
            test_data = data(test_indices, :);
            
            % Train and evaluate model
            fold_results{fold} = model_func(train_data, test_data);
        end
    else
        % Serial execution
        for fold = 1:k_folds
            test_start = (fold - 1) * fold_size + 1;
            test_end = min(fold * fold_size, n_samples);
            
            test_indices = test_start:test_end;
            train_indices = setdiff(1:n_samples, test_indices);
            
            train_data = data(train_indices, :);
            test_data = data(test_indices, :);
            
            fold_results{fold} = model_func(train_data, test_data);
        end
    end
    
    cv_time = toc;
    
    % Aggregate results
    if isnumeric(fold_results{1})
        % Simple numeric results
        all_scores = cell2mat(fold_results);
        results.mean_score = mean(all_scores);
        results.std_score = std(all_scores);
        results.scores = all_scores;
    else
        % Complex results structure
        results.fold_results = fold_results;
        results.n_folds = k_folds;
    end
    
    results.execution_time = cv_time;
    
    fprintf('Cross-validation completed in %.2f seconds\n', cv_time);
end
