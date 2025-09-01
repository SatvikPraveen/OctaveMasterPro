% File location: OctaveMasterPro/flagship_project/project_scripts/predictive_modeling.m
% Predictive modeling module for IoT predictive maintenance project

function model_results = predictive_modeling(processed_data)
    % Build and train predictive models for failure prediction
    % Input: processed_data structure with master_dataset
    % Returns: trained models and predictions
    
    fprintf('Starting predictive modeling...\n');
    
    % Initialize results structure
    model_results = struct();
    
    % Prepare data for modeling
    [X, y_7, y_30, feature_names, valid_indices] = prepare_modeling_data(processed_data.master_dataset);
    
    if size(X, 1) < 10
        fprintf('Warning: Insufficient data for modeling (n=%d)\n', size(X, 1));
        model_results.error = 'Insufficient data for modeling';
        return;
    end
    
    % Split data into training and testing sets
    [X_train, X_test, y_train_7, y_test_7, y_train_30, y_test_30, train_idx, test_idx] = ...
        split_data(X, y_7, y_30, 0.8);
    
    % Train models for 7-day prediction
    fprintf('Training 7-day failure prediction models...\n');
    model_results.models_7_day = train_models(X_train, y_train_7, feature_names);
    
    % Train models for 30-day prediction
    fprintf('Training 30-day failure prediction models...\n');
    model_results.models_30_day = train_models(X_train, y_train_30, feature_names);
    
    % Make predictions
    model_results.predictions_7_day = make_predictions(model_results.models_7_day, X_test, y_test_7);
    model_results.predictions_30_day = make_predictions(model_results.models_30_day, X_test, y_test_30);
    
    % Feature importance analysis
    model_results.feature_importance_7_day = analyze_model_feature_importance(model_results.models_7_day, feature_names);
    model_results.feature_importance_30_day = analyze_model_feature_importance(model_results.models_30_day, feature_names);
    
    % Store data splits for evaluation
    model_results.data_splits.X_train = X_train;
    model_results.data_splits.X_test = X_test;
    model_results.data_splits.y_train_7 = y_train_7;
    model_results.data_splits.y_test_7 = y_test_7;
    model_results.data_splits.y_train_30 = y_train_30;
    model_results.data_splits.y_test_30 = y_test_30;
    model_results.data_splits.feature_names = feature_names;
    model_results.data_splits.train_indices = train_idx;
    model_results.data_splits.test_indices = test_idx;
    
    % Save models
    save_models(model_results);
    
    fprintf('Predictive modeling completed\n');
end

function [X, y_7, y_30, feature_names, valid_indices] = prepare_modeling_data(master_dataset)
    % Prepare data for machine learning models
    
    fprintf('Preparing modeling data...\n');
    
    % Define feature columns
    potential_features = {'Avg_Temperature', 'Avg_Pressure', 'Avg_Humidity', 'Avg_Vibration', ...
                         'Max_Vibration', 'Std_Temperature', 'Sensor_Index', 'Age_Days', ...
                         'Operational_Hours', 'Historical_Failure_Rate', 'Maintenance_Efficiency', ...
                         'Days_Since_Maintenance'};
    
    % Check which features exist in the dataset
    available_features = intersect(potential_features, master_dataset.Properties.VariableNames);
    
    if isempty(available_features)
        error('No valid features found in master dataset');
    end
    
    % Extract feature matrix
    X_table = master_dataset(:, available_features);
    
    % Handle categorical variables (convert Equipment_Type if present)
    if any(strcmp(master_dataset.Properties.VariableNames, 'Equipment_Type'))
        equipment_dummies = dummyvar(categorical(master_dataset.Equipment_Type));
        equipment_types = categories(categorical(master_dataset.Equipment_Type));
        
        % Add dummy variables
        for i = 1:size(equipment_dummies, 2)
            var_name = ['EquipType_' matlab.lang.makeValidName(char(equipment_types{i}))];
            X_table.(var_name) = equipment_dummies(:, i);
            available_features{end+1} = var_name;
        end
    end
    
    % Convert to numeric matrix
    X = table2array(X_table(:, available_features));
    
    % Extract target variables
    if any(strcmp(master_dataset.Properties.VariableNames, 'Failure_Next_7_Days'))
        y_7 = double(master_dataset.Failure_Next_7_Days);
    else
        y_7 = zeros(height(master_dataset), 1);
    end
    
    if any(strcmp(master_dataset.Properties.VariableNames, 'Failure_Next_30_Days'))
        y_30 = double(master_dataset.Failure_Next_30_Days);
    else
        y_30 = zeros(height(master_dataset), 1);
    end
    
    % Find rows with complete data
    valid_indices = ~any(isnan(X), 2) & ~isnan(y_7) & ~isnan(y_30);
    
    X = X(valid_indices, :);
    y_7 = y_7(valid_indices);
    y_30 = y_30(valid_indices);
    
    feature_names = available_features;
    
    % Normalize features
    X = normalize(X, 1);
    
    fprintf('Prepared %d samples with %d features\n', size(X, 1), size(X, 2));
    fprintf('7-day failure rate: %.1f%%\n', mean(y_7) * 100);
    fprintf('30-day failure rate: %.1f%%\n', mean(y_30) * 100);
end

function [X_train, X_test, y_train_7, y_test_7, y_train_30, y_test_30, train_idx, test_idx] = ...
         split_data(X, y_7, y_30, train_ratio)
    % Split data into training and testing sets
    
    n_samples = size(X, 1);
    n_train = round(n_samples * train_ratio);
    
    % Random split (stratified by target if possible)
    rng(42); % for reproducibility
    
    % Try stratified split for 7-day target
    if sum(y_7) > 1 && sum(y_7) < (n_samples - 1)
        % Stratified sampling
        pos_indices = find(y_7 == 1);
        neg_indices = find(y_7 == 0);
        
        n_pos_train = round(length(pos_indices) * train_ratio);
        n_neg_train = n_train - n_pos_train;
        
        train_pos = pos_indices(randperm(length(pos_indices), min(n_pos_train, length(pos_indices))));
        train_neg = neg_indices(randperm(length(neg_indices), min(n_neg_train, length(neg_indices))));
        
        train_idx = sort([train_pos; train_neg]);
        test_idx = setdiff(1:n_samples, train_idx);
    else
        % Random split
        indices = randperm(n_samples);
        train_idx = sort(indices(1:n_train));
        test_idx = sort(indices(n_train+1:end));
    end
    
    % Split the data
    X_train = X(train_idx, :);
    X_test = X(test_idx, :);
    y_train_7 = y_7(train_idx);
    y_test_7 = y_7(test_idx);
    y_train_30 = y_30(train_idx);
    y_test_30 = y_30(test_idx);
    
    fprintf('Training set: %d samples (%.1f%% failures)\n', length(train_idx), mean(y_train_7)*100);
    fprintf('Test set: %d samples (%.1f%% failures)\n', length(test_idx), mean(y_test_7)*100);
end

function models = train_models(X_train, y_train, feature_names)
    % Train multiple models for failure prediction
    
    models = struct();
    
    if sum(y_train) == 0 || sum(y_train) == length(y_train)
        fprintf('Warning: No class variation in training data\n');
        models.error = 'No class variation in training data';
        return;
    end
    
    % 1. Logistic Regression (using iterative method)
    try
        fprintf('  Training logistic regression...\n');
        models.logistic = train_logistic_regression(X_train, y_train);
    catch ME
        fprintf('  Logistic regression failed: %s\n', ME.message);
        models.logistic.error = ME.message;
    end
    
    % 2. Decision Tree (simple implementation)
    try
        fprintf('  Training decision tree...\n');
        models.tree = train_decision_tree(X_train, y_train, feature_names);
    catch ME
        fprintf('  Decision tree failed: %s\n', ME.message);
        models.tree.error = ME.message;
    end
    
    % 3. Naive Bayes (Gaussian)
    try
        fprintf('  Training naive bayes...\n');
        models.naive_bayes = train_naive_bayes(X_train, y_train);
    catch ME
        fprintf('  Naive Bayes failed: %s\n', ME.message);
        models.naive_bayes.error = ME.message;
    end
    
    % 4. K-Nearest Neighbors
    try
        fprintf('  Training k-nearest neighbors...\n');
        models.knn = train_knn(X_train, y_train);
    catch ME
        fprintf('  KNN failed: %s\n', ME.message);
        models.knn.error = ME.message;
    end
    
    % 5. Ensemble method (simple voting)
    models.ensemble = create_ensemble(models);
end

function model = train_logistic_regression(X_train, y_train)
    % Simple logistic regression using gradient descent
    
    model = struct();
    model.type = 'logistic_regression';
    
    % Add bias term
    X_with_bias = [ones(size(X_train, 1), 1), X_train];
    
    % Initialize parameters
    beta = zeros(size(X_with_bias, 2), 1);
    
    % Hyperparameters
    learning_rate = 0.01;
    max_iterations = 1000;
    tolerance = 1e-6;
    
    % Gradient descent
    for iter = 1:max_iterations
        % Predictions
        z = X_with_bias * beta;
        predictions = 1 ./ (1 + exp(-z));
        
        % Cost function (log-likelihood)
        cost = -mean(y_train .* log(predictions + eps) + (1 - y_train) .* log(1 - predictions + eps));
        
        % Gradient
        gradient = X_with_bias' * (predictions - y_train) / length(y_train);
        
        % Update parameters
        beta_new = beta - learning_rate * gradient;
        
        % Check convergence
        if norm(beta_new - beta) < tolerance
            break;
        end
        
        beta = beta_new;
    end
    
    model.coefficients = beta;
    model.iterations = iter;
    
    % Model predictions
    z_train = X_with_bias * beta;
    train_predictions = 1 ./ (1 + exp(-z_train));
    model.train_accuracy = mean((train_predictions > 0.5) == y_train);
end

function model = train_decision_tree(X_train, y_train, feature_names)
    % Simple decision tree implementation
    
    model = struct();
    model.type = 'decision_tree';
    model.feature_names = feature_names;
    
    % Build tree recursively
    model.tree = build_tree(X_train, y_train, feature_names, 1, 5); % max_depth = 5
    
    % Calculate training accuracy
    train_predictions = predict_tree(model.tree, X_train);
    model.train_accuracy = mean(train_predictions == y_train);
end

function tree = build_tree(X, y, feature_names, depth, max_depth)
    % Recursive tree building
    
    tree = struct();
    
    % Stopping criteria
    if depth > max_depth || length(unique(y)) == 1 || size(X, 1) < 5
        tree.is_leaf = true;
        tree.prediction = mode(y);
        tree.probability = mean(y);
        return;
    end
    
    tree.is_leaf = false;
    
    % Find best split
    best_gini = inf;
    best_feature = 1;
    best_threshold = 0;
    
    for feature_idx = 1:size(X, 2)
        feature_values = unique(X(:, feature_idx));
        
        for threshold_idx = 1:min(10, length(feature_values))
            threshold = feature_values(threshold_idx);
            
            left_mask = X(:, feature_idx) <= threshold;
            right_mask = ~left_mask;
            
            if sum(left_mask) == 0 || sum(right_mask) == 0
                continue;
            end
            
            % Calculate weighted Gini impurity
            left_gini = calculate_gini(y(left_mask));
            right_gini = calculate_gini(y(right_mask));
            
            weighted_gini = (sum(left_mask) * left_gini + sum(right_mask) * right_gini) / length(y);
            
            if weighted_gini < best_gini
                best_gini = weighted_gini;
                best_feature = feature_idx;
                best_threshold = threshold;
            end
        end
    end
    
    % Split data
    left_mask = X(:, best_feature) <= best_threshold;
    right_mask = ~left_mask;
    
    tree.feature_idx = best_feature;
    tree.threshold = best_threshold;
    tree.feature_name = feature_names{best_feature};
    
    % Build child trees
    tree.left = build_tree(X(left_mask, :), y(left_mask), feature_names, depth + 1, max_depth);
    tree.right = build_tree(X(right_mask, :), y(right_mask), feature_names, depth + 1, max_depth);
end

function gini = calculate_gini(y)
    % Calculate Gini impurity
    if isempty(y)
        gini = 0;
        return;
    end
    
    classes = unique(y);
    gini = 1;
    
    for i = 1:length(classes)
        p = sum(y == classes(i)) / length(y);
        gini = gini - p^2;
    end
end

function predictions = predict_tree(tree, X)
    % Make predictions using decision tree
    
    predictions = zeros(size(X, 1), 1);
    
    for i = 1:size(X, 1)
        predictions(i) = predict_single_tree(tree, X(i, :));
    end
end

function prediction = predict_single_tree(tree, x)
    % Predict single sample using decision tree
    
    if tree.is_leaf
        prediction = tree.prediction;
        return;
    end
    
    if x(tree.feature_idx) <= tree.threshold
        prediction = predict_single_tree(tree.left, x);
    else
        prediction = predict_single_tree(tree.right, x);
    end
end

function model = train_naive_bayes(X_train, y_train)
    % Gaussian Naive Bayes implementation
    
    model = struct();
    model.type = 'naive_bayes';
    
    classes = unique(y_train);
    model.classes = classes;
    model.class_priors = zeros(length(classes), 1);
    model.feature_means = zeros(length(classes), size(X_train, 2));
    model.feature_stds = zeros(length(classes), size(X_train, 2));
    
    for i = 1:length(classes)
        class_mask = (y_train == classes(i));
        class_data = X_train(class_mask, :);
        
        model.class_priors(i) = sum(class_mask) / length(y_train);
        model.feature_means(i, :) = mean(class_data, 1);
        model.feature_stds(i, :) = std(class_data, 1) + 1e-6; % Add small constant for numerical stability
    end
    
    % Training accuracy
    train_predictions = predict_naive_bayes(model, X_train);
    model.train_accuracy = mean(train_predictions == y_train);
end

function predictions = predict_naive_bayes(model, X)
    % Make predictions using Naive Bayes
    
    predictions = zeros(size(X, 1), 1);
    
    for i = 1:size(X, 1)
        log_probs = zeros(length(model.classes), 1);
        
        for j = 1:length(model.classes)
            log_probs(j) = log(model.class_priors(j));
            
            for k = 1:size(X, 2)
                % Gaussian likelihood
                mean_val = model.feature_means(j, k);
                std_val = model.feature_stds(j, k);
                
                log_likelihood = -0.5 * log(2 * pi * std_val^2) - ...
                                0.5 * ((X(i, k) - mean_val) / std_val)^2;
                log_probs(j) = log_probs(j) + log_likelihood;
            end
        end
        
        [~, max_idx] = max(log_probs);
        predictions(i) = model.classes(max_idx);
    end
end

function model = train_knn(X_train, y_train)
    % K-Nearest Neighbors implementation
    
    model = struct();
    model.type = 'knn';
    model.X_train = X_train;
    model.y_train = y_train;
    model.k = min(5, floor(size(X_train, 1) / 2)); % k=5 or half of training data
    
    % Training accuracy (using leave-one-out for efficiency)
    correct = 0;
    for i = 1:min(50, size(X_train, 1)) % Sample for efficiency
        % Find k nearest neighbors (excluding self)
        distances = sqrt(sum((X_train - X_train(i, :)).^2, 2));
        distances(i) = inf; % Exclude self
        
        [~, nearest_idx] = sort(distances);
        k_nearest = nearest_idx(1:model.k);
        
        prediction = mode(y_train(k_nearest));
        if prediction == y_train(i)
            correct = correct + 1;
        end
    end
    
    model.train_accuracy = correct / min(50, size(X_train, 1));
end

function predictions = predict_knn(model, X_test)
    % Make predictions using KNN
    
    predictions = zeros(size(X_test, 1), 1);
    
    for i = 1:size(X_test, 1)
        % Calculate distances to all training points
        distances = sqrt(sum((model.X_train - X_test(i, :)).^2, 2));
        
        % Find k nearest neighbors
        [~, nearest_idx] = sort(distances);
        k_nearest = nearest_idx(1:model.k);
        
        % Majority vote
        predictions(i) = mode(model.y_train(k_nearest));
    end
end

function ensemble = create_ensemble(models)
    % Create ensemble model from individual models
    
    ensemble = struct();
    ensemble.type = 'ensemble';
    ensemble.models = models;
    
    % Remove failed models
    model_names = fieldnames(models);
    valid_models = {};
    
    for i = 1:length(model_names)
        if ~isfield(models.(model_names{i}), 'error')
            valid_models{end+1} = model_names{i};
        end
    end
    
    ensemble.valid_models = valid_models;
    
    if isempty(valid_models)
        ensemble.error = 'No valid models for ensemble';
    end
end

function predictions = make_predictions(models, X_test, y_test)
    % Make predictions using trained models
    
    predictions = struct();
    model_names = fieldnames(models);
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        model = models.(model_name);
        
        if isfield(model, 'error')
            continue;
        end
        
        try
            switch model.type
                case 'logistic_regression'
                    X_with_bias = [ones(size(X_test, 1), 1), X_test];
                    z = X_with_bias * model.coefficients;
                    probs = 1 ./ (1 + exp(-z));
                    preds = double(probs > 0.5);
                    
                case 'decision_tree'
                    preds = predict_tree(model.tree, X_test);
                    probs = preds; % Binary predictions as probabilities
                    
                case 'naive_bayes'
                    preds = predict_naive_bayes(model, X_test);
                    probs = preds; % Binary predictions as probabilities
                    
                case 'knn'
                    preds = predict_knn(model, X_test);
                    probs = preds; % Binary predictions as probabilities
                    
                case 'ensemble'
                    if ~isfield(model, 'error')
                        [preds, probs] = predict_ensemble(model, X_test);
                    else
                        continue;
                    end
                    
                otherwise
                    continue;
            end
            
            predictions.(model_name).predictions = preds;
            predictions.(model_name).probabilities = probs;
            predictions.(model_name).accuracy = mean(preds == y_test);
            
        catch ME
            fprintf('Prediction failed for %s: %s\n', model_name, ME.message);
        end
    end
end

function [predictions, probabilities] = predict_ensemble(ensemble_model, X_test)
    % Make ensemble predictions using voting
    
    if isfield(ensemble_model, 'error')
        predictions = zeros(size(X_test, 1), 1);
        probabilities = zeros(size(X_test, 1), 1);
        return;
    end
    
    valid_models = ensemble_model.valid_models;
    n_models = length(valid_models);
    
    if n_models == 0
        predictions = zeros(size(X_test, 1), 1);
        probabilities = zeros(size(X_test, 1), 1);
        return;
    end
    
    all_predictions = zeros(size(X_test, 1), n_models);
    
    for i = 1:n_models
        model_name = valid_models{i};
        model = ensemble_model.models.(model_name);
        
        switch model.type
            case 'logistic_regression'
                X_with_bias = [ones(size(X_test, 1), 1), X_test];
                z = X_with_bias * model.coefficients;
                probs = 1 ./ (1 + exp(-z));
                all_predictions(:, i) = double(probs > 0.5);
                
            case 'decision_tree'
                all_predictions(:, i) = predict_tree(model.tree, X_test);
                
            case 'naive_bayes'
                all_predictions(:, i) = predict_naive_bayes(model, X_test);
                
            case 'knn'
                all_predictions(:, i) = predict_knn(model, X_test);
        end
    end
    
    % Majority voting
    predictions = mode(all_predictions, 2);
    probabilities = mean(all_predictions, 2);
end

function feature_importance = analyze_model_feature_importance(models, feature_names)
    % Analyze feature importance from trained models
    
    feature_importance = struct();
    model_names = fieldnames(models);
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        model = models.(model_name);
        
        if isfield(model, 'error')
            continue;
        end
        
        switch model.type
            case 'logistic_regression'
                % Use absolute coefficients as importance
                coeffs = abs(model.coefficients(2:end)); % Exclude bias term
                importance = coeffs / sum(coeffs);
                
                feature_importance.(model_name) = containers.Map(feature_names, importance);
                
            case 'decision_tree'
                % Count feature usage in tree
                importance = calculate_tree_importance(model.tree, length(feature_names));
                importance = importance / sum(importance);
                
                feature_importance.(model_name) = containers.Map(feature_names, importance);
                
            otherwise
                % Default uniform importance
                uniform_importance = ones(length(feature_names), 1) / length(feature_names);
                feature_importance.(model_name) = containers.Map(feature_names, uniform_importance);
        end
    end
end

function importance = calculate_tree_importance(tree, n_features)
    % Calculate feature importance for decision tree
    
    importance = zeros(n_features, 1);
    
    if tree.is_leaf
        return;
    end
    
    importance(tree.feature_idx) = importance(tree.feature_idx) + 1;
    
    left_importance = calculate_tree_importance(tree.left, n_features);
    right_importance = calculate_tree_importance(tree.right, n_features);
    
    importance = importance + left_importance + right_importance;
end

function save_models(model_results)
    % Save trained models to file
    
    fprintf('Saving models...\n');
    
    if ~exist('report/models', 'dir')
        mkdir('report/models');
    end
    
    % Save model results
    save('report/models/predictive_model.mat', 'model_results');
    
    % Save feature importance as CSV
    if isfield(model_results, 'feature_importance_7_day')
        feature_names = model_results.data_splits.feature_names;
        
        % Extract logistic regression importance if available
        if isfield(model_results.feature_importance_7_day, 'logistic')
            importance_map = model_results.feature_importance_7_day.logistic;
            importance_values = zeros(length(feature_names), 1);
            
            for i = 1:length(feature_names)
                if isKey(importance_map, feature_names{i})
                    importance_values(i) = importance_map(feature_names{i});
                end
            end
            
            % Create table and save
            importance_table = table(feature_names', importance_values, ...
                                   'VariableNames', {'Feature', 'Importance'});
            writetable(importance_table, 'report/models/feature_importance.csv');
        end
    end
    
    fprintf('Models saved to report/models/\n');
end