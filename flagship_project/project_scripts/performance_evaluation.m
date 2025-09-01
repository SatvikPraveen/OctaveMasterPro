% File location: OctaveMasterPro/flagship_project/project_scripts/performance_evaluation.m
% Performance evaluation module for IoT predictive maintenance project

function evaluation_results = performance_evaluation(model_results)
    % Comprehensive model performance evaluation
    % Input: model_results structure from predictive_modeling
    % Returns: evaluation metrics and performance visualizations
    
    fprintf('Starting performance evaluation...\n');
    
    if isfield(model_results, 'error')
        fprintf('Error: %s\n', model_results.error);
        evaluation_results.error = model_results.error;
        return;
    end
    
    % Initialize results structure
    evaluation_results = struct();
    
    % Evaluate 7-day prediction models
    if isfield(model_results, 'predictions_7_day') && ~isempty(model_results.predictions_7_day)
        fprintf('Evaluating 7-day prediction models...\n');
        evaluation_results.metrics_7_day = evaluate_models(model_results.predictions_7_day, ...
                                                           model_results.data_splits.y_test_7, '7-day');
    end
    
    % Evaluate 30-day prediction models
    if isfield(model_results, 'predictions_30_day') && ~isempty(model_results.predictions_30_day)
        fprintf('Evaluating 30-day prediction models...\n');
        evaluation_results.metrics_30_day = evaluate_models(model_results.predictions_30_day, ...
                                                            model_results.data_splits.y_test_30, '30-day');
    end
    
    % Model comparison
    evaluation_results.model_comparison = compare_models(evaluation_results);
    
    % Cross-validation if sufficient data
    if size(model_results.data_splits.X_train, 1) > 50
        fprintf('Performing cross-validation...\n');
        evaluation_results.cross_validation = perform_cross_validation(model_results);
    end
    
    % Business impact analysis
    evaluation_results.business_impact = calculate_business_impact(evaluation_results, model_results);
    
    % Generate performance visualizations
    generate_performance_plots(evaluation_results, model_results);
    
    % Save validation results
    save_validation_results(evaluation_results);
    
    fprintf('Performance evaluation completed\n');
end

function metrics = evaluate_models(predictions, y_true, prediction_type)
    % Calculate comprehensive evaluation metrics for all models
    
    metrics = struct();
    model_names = fieldnames(predictions);
    
    fprintf('  Calculating metrics for %s prediction...\n', prediction_type);
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        
        if ~isfield(predictions.(model_name), 'predictions')
            continue;
        end
        
        y_pred = predictions.(model_name).predictions;
        y_prob = predictions.(model_name).probabilities;
        
        % Basic classification metrics
        model_metrics = calculate_classification_metrics(y_true, y_pred, y_prob);
        
        % Additional metrics
        model_metrics.model_name = model_name;
        model_metrics.prediction_type = prediction_type;
        
        metrics.(model_name) = model_metrics;
        
        fprintf('    %s: Accuracy=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f\n', ...
                model_name, model_metrics.accuracy, model_metrics.precision, ...
                model_metrics.recall, model_metrics.f1_score);
    end
end

function metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    % Calculate standard classification metrics
    
    metrics = struct();
    
    % Confusion matrix components
    tp = sum(y_true == 1 & y_pred == 1);
    tn = sum(y_true == 0 & y_pred == 0);
    fp = sum(y_true == 0 & y_pred == 1);
    fn = sum(y_true == 1 & y_pred == 0);
    
    metrics.confusion_matrix = [tn, fp; fn, tp];
    metrics.true_positives = tp;
    metrics.true_negatives = tn;
    metrics.false_positives = fp;
    metrics.false_negatives = fn;
    
    % Basic metrics
    metrics.accuracy = (tp + tn) / (tp + tn + fp + fn);
    
    if (tp + fp) > 0
        metrics.precision = tp / (tp + fp);
    else
        metrics.precision = 0;
    end
    
    if (tp + fn) > 0
        metrics.recall = tp / (tp + fn);
        metrics.sensitivity = metrics.recall;
    else
        metrics.recall = 0;
        metrics.sensitivity = 0;
    end
    
    if (tn + fp) > 0
        metrics.specificity = tn / (tn + fp);
    else
        metrics.specificity = 0;
    end
    
    % F1 Score
    if (metrics.precision + metrics.recall) > 0
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
    else
        metrics.f1_score = 0;
    end
    
    % Matthews Correlation Coefficient
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    if denominator > 0
        metrics.mcc = (tp * tn - fp * fn) / denominator;
    else
        metrics.mcc = 0;
    end
    
    % ROC AUC (Area Under ROC Curve)
    if length(unique(y_true)) > 1 && length(unique(y_prob)) > 1
        metrics.roc_auc = calculate_roc_auc(y_true, y_prob);
        [metrics.roc_fpr, metrics.roc_tpr, metrics.roc_thresholds] = calculate_roc_curve(y_true, y_prob);
    else
        metrics.roc_auc = 0.5;
        metrics.roc_fpr = [0, 1];
        metrics.roc_tpr = [0, 1];
        metrics.roc_thresholds = [1, 0];
    end
    
    % Precision-Recall AUC
    if sum(y_true) > 0
        metrics.pr_auc = calculate_pr_auc(y_true, y_prob);
        [metrics.pr_precision, metrics.pr_recall, metrics.pr_thresholds] = calculate_pr_curve(y_true, y_prob);
    else
        metrics.pr_auc = 0;
        metrics.pr_precision = [1, 0];
        metrics.pr_recall = [0, 1];
        metrics.pr_thresholds = [0, 1];
    end
    
    % Log Loss
    y_prob_clamped = max(min(y_prob, 1-eps), eps); % Clamp probabilities
    metrics.log_loss = -mean(y_true .* log(y_prob_clamped) + (1-y_true) .* log(1-y_prob_clamped));
    
    % Balanced accuracy
    metrics.balanced_accuracy = (metrics.sensitivity + metrics.specificity) / 2;
end

function auc = calculate_roc_auc(y_true, y_scores)
    % Calculate Area Under ROC Curve using trapezoidal rule
    
    [fpr, tpr, ~] = calculate_roc_curve(y_true, y_scores);
    
    % Sort by FPR for proper integration
    [fpr_sorted, sort_idx] = sort(fpr);
    tpr_sorted = tpr(sort_idx);
    
    % Calculate AUC using trapezoidal rule
    auc = trapz(fpr_sorted, tpr_sorted);
end

function [fpr, tpr, thresholds] = calculate_roc_curve(y_true, y_scores)
    % Calculate ROC curve points
    
    unique_scores = unique(y_scores);
    thresholds = [inf; unique_scores; -inf];
    
    fpr = zeros(length(thresholds), 1);
    tpr = zeros(length(thresholds), 1);
    
    n_pos = sum(y_true == 1);
    n_neg = sum(y_true == 0);
    
    for i = 1:length(thresholds)
        y_pred = double(y_scores >= thresholds(i));
        
        tp = sum(y_true == 1 & y_pred == 1);
        fp = sum(y_true == 0 & y_pred == 1);
        
        if n_pos > 0
            tpr(i) = tp / n_pos;
        end
        
        if n_neg > 0
            fpr(i) = fp / n_neg;
        end
    end
end

function auc = calculate_pr_auc(y_true, y_scores)
    % Calculate Area Under Precision-Recall Curve
    
    [precision, recall, ~] = calculate_pr_curve(y_true, y_scores);
    
    % Sort by recall for proper integration
    [recall_sorted, sort_idx] = sort(recall);
    precision_sorted = precision(sort_idx);
    
    % Calculate AUC using trapezoidal rule
    auc = trapz(recall_sorted, precision_sorted);
end

function [precision, recall, thresholds] = calculate_pr_curve(y_true, y_scores)
    % Calculate Precision-Recall curve points
    
    unique_scores = unique(y_scores);
    thresholds = [inf; unique_scores; -inf];
    
    precision = zeros(length(thresholds), 1);
    recall = zeros(length(thresholds), 1);
    
    n_pos = sum(y_true == 1);
    
    for i = 1:length(thresholds)
        y_pred = double(y_scores >= thresholds(i));
        
        tp = sum(y_true == 1 & y_pred == 1);
        fp = sum(y_true == 0 & y_pred == 1);
        
        if (tp + fp) > 0
            precision(i) = tp / (tp + fp);
        else
            precision(i) = 1;
        end
        
        if n_pos > 0
            recall(i) = tp / n_pos;
        end
    end
end

function comparison = compare_models(evaluation_results)
    % Compare model performances and identify best models
    
    comparison = struct();
    
    % Compare 7-day models
    if isfield(evaluation_results, 'metrics_7_day')
        comparison.best_7_day = find_best_model(evaluation_results.metrics_7_day);
        comparison.ranking_7_day = rank_models(evaluation_results.metrics_7_day);
    end
    
    % Compare 30-day models
    if isfield(evaluation_results, 'metrics_30_day')
        comparison.best_30_day = find_best_model(evaluation_results.metrics_30_day);
        comparison.ranking_30_day = rank_models(evaluation_results.metrics_30_day);
    end
    
    % Overall comparison summary
    if isfield(comparison, 'best_7_day') && isfield(comparison, 'best_30_day')
        comparison.summary = sprintf('Best 7-day model: %s (F1=%.3f), Best 30-day model: %s (F1=%.3f)', ...
                                   comparison.best_7_day.model, comparison.best_7_day.f1_score, ...
                                   comparison.best_30_day.model, comparison.best_30_day.f1_score);
        
        fprintf('  %s\n', comparison.summary);
    end
end

function best_model = find_best_model(metrics)
    % Find best performing model based on F1 score
    
    model_names = fieldnames(metrics);
    best_f1 = -1;
    best_model = struct();
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        
        if isfield(metrics.(model_name), 'f1_score')
            f1 = metrics.(model_name).f1_score;
            
            if f1 > best_f1
                best_f1 = f1;
                best_model.model = model_name;
                best_model.f1_score = f1;
                best_model.accuracy = metrics.(model_name).accuracy;
                best_model.precision = metrics.(model_name).precision;
                best_model.recall = metrics.(model_name).recall;
                best_model.roc_auc = metrics.(model_name).roc_auc;
            end
        end
    end
end

function ranking = rank_models(metrics)
    % Rank all models by multiple metrics
    
    model_names = fieldnames(metrics);
    ranking = struct();
    
    if isempty(model_names)
        return;
    end
    
    % Extract metrics for ranking
    f1_scores = zeros(length(model_names), 1);
    accuracies = zeros(length(model_names), 1);
    roc_aucs = zeros(length(model_names), 1);
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        
        if isfield(metrics.(model_name), 'f1_score')
            f1_scores(i) = metrics.(model_name).f1_score;
            accuracies(i) = metrics.(model_name).accuracy;
            roc_aucs(i) = metrics.(model_name).roc_auc;
        end
    end
    
    % Composite score (weighted average)
    composite_scores = 0.5 * f1_scores + 0.3 * accuracies + 0.2 * roc_aucs;
    
    [sorted_scores, sort_idx] = sort(composite_scores, 'descend');
    
    ranking.model_order = model_names(sort_idx);
    ranking.composite_scores = sorted_scores;
    ranking.f1_scores = f1_scores(sort_idx);
    ranking.accuracies = accuracies(sort_idx);
    ranking.roc_aucs = roc_aucs(sort_idx);
end

function cv_results = perform_cross_validation(model_results)
    % Perform k-fold cross-validation
    
    fprintf('  Running 5-fold cross-validation...\n');
    
    cv_results = struct();
    k_folds = 5;
    
    X = [model_results.data_splits.X_train; model_results.data_splits.X_test];
    y_7 = [model_results.data_splits.y_train_7; model_results.data_splits.y_test_7];
    y_30 = [model_results.data_splits.y_train_30; model_results.data_splits.y_test_30];
    
    n_samples = size(X, 1);
    fold_size = floor(n_samples / k_folds);
    
    % Initialize results storage
    cv_accuracies_7 = [];
    cv_f1_scores_7 = [];
    cv_accuracies_30 = [];
    cv_f1_scores_30 = [];
    
    for fold = 1:k_folds
        fprintf('    Fold %d/%d...\n', fold, k_folds);
        
        % Define test indices for this fold
        test_start = (fold - 1) * fold_size + 1;
        test_end = min(fold * fold_size, n_samples);
        test_idx = test_start:test_end;
        train_idx = setdiff(1:n_samples, test_idx);
        
        % Split data
        X_train_fold = X(train_idx, :);
        X_test_fold = X(test_idx, :);
        y_train_7_fold = y_7(train_idx);
        y_test_7_fold = y_7(test_idx);
        y_train_30_fold = y_30(train_idx);
        y_test_30_fold = y_30(test_idx);
        
        % Train simple logistic regression for CV
        try
            % 7-day model
            model_7 = train_simple_logistic(X_train_fold, y_train_7_fold);
            if ~isfield(model_7, 'error')
                pred_7 = predict_simple_logistic(model_7, X_test_fold);
                cv_accuracies_7(end+1) = mean(pred_7 == y_test_7_fold);
                
                % Calculate F1 score
                tp = sum(y_test_7_fold == 1 & pred_7 == 1);
                fp = sum(y_test_7_fold == 0 & pred_7 == 1);
                fn = sum(y_test_7_fold == 1 & pred_7 == 0);
                
                if (tp + fp) > 0 && (tp + fn) > 0
                    precision = tp / (tp + fp);
                    recall = tp / (tp + fn);
                    f1 = 2 * precision * recall / (precision + recall);
                else
                    f1 = 0;
                end
                cv_f1_scores_7(end+1) = f1;
            end
            
            % 30-day model
            model_30 = train_simple_logistic(X_train_fold, y_train_30_fold);
            if ~isfield(model_30, 'error')
                pred_30 = predict_simple_logistic(model_30, X_test_fold);
                cv_accuracies_30(end+1) = mean(pred_30 == y_test_30_fold);
                
                % Calculate F1 score
                tp = sum(y_test_30_fold == 1 & pred_30 == 1);
                fp = sum(y_test_30_fold == 0 & pred_30 == 1);
                fn = sum(y_test_30_fold == 1 & pred_30 == 0);
                
                if (tp + fp) > 0 && (tp + fn) > 0
                    precision = tp / (tp + fp);
                    recall = tp / (tp + fn);
                    f1 = 2 * precision * recall / (precision + recall);
                else
                    f1 = 0;
                end
                cv_f1_scores_30(end+1) = f1;
            end
            
        catch ME
            fprintf('    Fold %d failed: %s\n', fold, ME.message);
        end
    end
    
    % Summarize CV results
    if ~isempty(cv_accuracies_7)
        cv_results.cv_7_day.mean_accuracy = mean(cv_accuracies_7);
        cv_results.cv_7_day.std_accuracy = std(cv_accuracies_7);
        cv_results.cv_7_day.mean_f1 = mean(cv_f1_scores_7);
        cv_results.cv_7_day.std_f1 = std(cv_f1_scores_7);
        
        fprintf('    7-day CV Results: Accuracy = %.3f ± %.3f, F1 = %.3f ± %.3f\n', ...
                cv_results.cv_7_day.mean_accuracy, cv_results.cv_7_day.std_accuracy, ...
                cv_results.cv_7_day.mean_f1, cv_results.cv_7_day.std_f1);
    end
    
    if ~isempty(cv_accuracies_30)
        cv_results.cv_30_day.mean_accuracy = mean(cv_accuracies_30);
        cv_results.cv_30_day.std_accuracy = std(cv_accuracies_30);
        cv_results.cv_30_day.mean_f1 = mean(cv_f1_scores_30);
        cv_results.cv_30_day.std_f1 = std(cv_f1_scores_30);
        
        fprintf('    30-day CV Results: Accuracy = %.3f ± %.3f, F1 = %.3f ± %.3f\n', ...
                cv_results.cv_30_day.mean_accuracy, cv_results.cv_30_day.std_accuracy, ...
                cv_results.cv_30_day.mean_f1, cv_results.cv_30_day.std_f1);
    end
end

function model = train_simple_logistic(X, y)
    % Simple logistic regression for cross-validation
    
    model = struct();
    
    if sum(y) == 0 || sum(y) == length(y)
        model.error = 'No class variation';
        return;
    end
    
    % Add bias term
    X_with_bias = [ones(size(X, 1), 1), X];
    
    % Simple gradient descent
    beta = zeros(size(X_with_bias, 2), 1);
    learning_rate = 0.01;
    
    for iter = 1:100
        z = X_with_bias * beta;
        predictions = 1 ./ (1 + exp(-z));
        gradient = X_with_bias' * (predictions - y) / length(y);
        beta = beta - learning_rate * gradient;
    end
    
    model.coefficients = beta;
end

function predictions = predict_simple_logistic(model, X)
    % Make predictions with simple logistic model
    
    X_with_bias = [ones(size(X, 1), 1), X];
    z = X_with_bias * model.coefficients;
    probabilities = 1 ./ (1 + exp(-z));
    predictions = double(probabilities > 0.5);
end

function business_impact = calculate_business_impact(evaluation_results, model_results)
    % Calculate business impact metrics
    
    business_impact = struct();
    
    % Assumptions for cost-benefit analysis
    cost_per_failure = 10000; % Average cost of unplanned failure
    cost_per_false_positive = 500; % Cost of unnecessary maintenance
    maintenance_cost_savings = 0.3; % 30% reduction in maintenance costs
    
    % Calculate impact for best 7-day model
    if isfield(evaluation_results, 'metrics_7_day') && ...
       isfield(evaluation_results, 'model_comparison') && ...
       isfield(evaluation_results.model_comparison, 'best_7_day')
        
        best_7_day = evaluation_results.model_comparison.best_7_day;
        best_metrics = evaluation_results.metrics_7_day.(best_7_day.model);
        
        % Calculate cost savings
        tp = best_metrics.true_positives;
        fp = best_metrics.false_positives;
        fn = best_metrics.false_negatives;
        
        prevented_failures_cost = tp * cost_per_failure;
        false_positive_cost = fp * cost_per_false_positive;
        missed_failures_cost = fn * cost_per_failure;
        
        net_savings = prevented_failures_cost - false_positive_cost - missed_failures_cost;
        
        business_impact.model_7_day.prevented_failures = tp;
        business_impact.model_7_day.false_alarms = fp;
        business_impact.model_7_day.missed_failures = fn;
        business_impact.model_7_day.cost_savings = net_savings;
        business_impact.model_7_day.roi_percentage = (net_savings / (prevented_failures_cost + false_positive_cost)) * 100;
        
        fprintf('  7-day model business impact: $%.0f savings, %.1f%% ROI\n', ...
                net_savings, business_impact.model_7_day.roi_percentage);
    end
    
    % Calculate impact for best 30-day model
    if isfield(evaluation_results, 'metrics_30_day') && ...
       isfield(evaluation_results, 'model_comparison') && ...
       isfield(evaluation_results.model_comparison, 'best_30_day')
        
        best_30_day = evaluation_results.model_comparison.best_30_day;
        best_metrics = evaluation_results.metrics_30_day.(best_30_day.model);
        
        % Calculate cost savings
        tp = best_metrics.true_positives;
        fp = best_metrics.false_positives;
        fn = best_metrics.false_negatives;
        
        prevented_failures_cost = tp * cost_per_failure;
        false_positive_cost = fp * cost_per_false_positive;
        missed_failures_cost = fn * cost_per_failure;
        
        net_savings = prevented_failures_cost - false_positive_cost - missed_failures_cost;
        
        business_impact.model_30_day.prevented_failures = tp;
        business_impact.model_30_day.false_alarms = fp;
        business_impact.model_30_day.missed_failures = fn;
        business_impact.model_30_day.cost_savings = net_savings;
        business_impact.model_30_day.roi_percentage = (net_savings / (prevented_failures_cost + false_positive_cost)) * 100;
        
        fprintf('  30-day model business impact: $%.0f savings, %.1f%% ROI\n', ...
                net_savings, business_impact.model_30_day.roi_percentage);
    end
end

function generate_performance_plots(evaluation_results, model_results)
    % Generate performance visualization plots
    
    fprintf('Generating performance plots...\n');
    
    if ~exist('report/figures', 'dir')
        mkdir('report/figures');
    end
    
    % ROC Curves
    if isfield(evaluation_results, 'metrics_7_day')
        plot_roc_curves(evaluation_results.metrics_7_day, '7-day Prediction ROC Curves');
        saveas(gcf, 'report/figures/roc_curves_7day.png');
        close;
    end
    
    if isfield(evaluation_results, 'metrics_30_day')
        plot_roc_curves(evaluation_results.metrics_30_day, '30-day Prediction ROC Curves');
        saveas(gcf, 'report/figures/roc_curves_30day.png');
        close;
    end
    
    % Performance comparison
    plot_performance_comparison(evaluation_results);
    saveas(gcf, 'report/figures/performance_metrics.png');
    close;
    
    fprintf('Performance plots saved to report/figures/\n');
end

function plot_roc_curves(metrics, title_str)
    % Plot ROC curves for all models
    
    figure('Position', [100, 100, 800, 600]);
    hold on;
    
    model_names = fieldnames(metrics);
    colors = lines(length(model_names));
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        
        if isfield(metrics.(model_name), 'roc_fpr')
            fpr = metrics.(model_name).roc_fpr;
            tpr = metrics.(model_name).roc_tpr;
            auc = metrics.(model_name).roc_auc;
            
            plot(fpr, tpr, 'Color', colors(i,:), 'LineWidth', 2, ...
                 'DisplayName', sprintf('%s (AUC=%.3f)', model_name, auc));
        end
    end
    
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1, 'DisplayName', 'Random');
    
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(title_str);
    legend('Location', 'southeast');
    grid on;
    hold off;
end

function plot_performance_comparison(evaluation_results)
    % Plot performance metrics comparison
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Collect metrics for plotting
    model_names_7 = {};
    accuracies_7 = [];
    f1_scores_7 = [];
    
    if isfield(evaluation_results, 'metrics_7_day')
        fields = fieldnames(evaluation_results.metrics_7_day);
        for i = 1:length(fields)
            if isfield(evaluation_results.metrics_7_day.(fields{i}), 'accuracy')
                model_names_7{end+1} = fields{i};
                accuracies_7(end+1) = evaluation_results.metrics_7_day.(fields{i}).accuracy;
                f1_scores_7(end+1) = evaluation_results.metrics_7_day.(fields{i}).f1_score;
            end
        end
    end
    
    if ~isempty(model_names_7)
        subplot(2, 1, 1);
        x = 1:length(model_names_7);
        
        bar_width = 0.35;
        bar(x - bar_width/2, accuracies_7, bar_width, 'DisplayName', 'Accuracy');
        hold on;
        bar(x + bar_width/2, f1_scores_7, bar_width, 'DisplayName', 'F1 Score');
        
        set(gca, 'XTick', x);
        set(gca, 'XTickLabel', model_names_7);
        xlabel('Models');
        ylabel('Score');
        title('7-day Prediction Performance');
        legend();
        grid on;
        hold off;
    end
    
    % Similar for 30-day models
    model_names_30 = {};
    accuracies_30 = [];
    f1_scores_30 = [];
    
    if isfield(evaluation_results, 'metrics_30_day')
        fields = fieldnames(evaluation_results.metrics_30_day);
        for i = 1:length(fields)
            if isfield(evaluation_results.metrics_30_day.(fields{i}), 'accuracy')
                model_names_30{end+1} = fields{i};
                accuracies_30(end+1) = evaluation_results.metrics_30_day.(fields{i}).accuracy;
                f1_scores_30(end+1) = evaluation_results.metrics_30_day.(fields{i}).f1_score;
            end
        end
    end
    
    if ~isempty(model_names_30)
        subplot(2, 1, 2);
        x = 1:length(model_names_30);
        
        bar_width = 0.35;
        bar(x - bar_width/2, accuracies_30, bar_width, 'DisplayName', 'Accuracy');
        hold on;
        bar(x + bar_width/2, f1_scores_30, bar_width, 'DisplayName', 'F1 Score');
        
        set(gca, 'XTick', x);
        set(gca, 'XTickLabel', model_names_30);
        xlabel('Models');
        ylabel('Score');
        title('30-day Prediction Performance');
        legend();
        grid on;
        hold off;
    end
end

function save_validation_results(evaluation_results)
    % Save validation results to file
    
    fprintf('Saving validation results...\n');
    
    if ~exist('report/models', 'dir')
        mkdir('report/models');
    end
    
    save('report/models/validation_results.mat', 'evaluation_results');
    
    fprintf('Validation results saved to report/models/\n');
end