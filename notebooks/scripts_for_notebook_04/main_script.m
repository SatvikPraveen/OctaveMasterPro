% File: notebooks/scripts_for_notebook_04/main_script.m
% Main script demonstrating script organization and function calls

% Clear workspace and command window
clear all;
clc;

fprintf('=== OctaveMasterPro: Main Script Demo ===\n');
fprintf('This script demonstrates script organization and function usage.\n\n');

% Define global parameters
global_param = 42;
data_size = 100;

% Generate sample data
fprintf('1. Generating sample data...\n');
x = linspace(0, 2*pi, data_size);
y = sin(x) + 0.1 * randn(size(x));

% Call external script
fprintf('2. Running external script (script_zero.m)...\n');
run('script_zero.m');

% Define and use local functions
fprintf('3. Using local functions...\n');

% Calculate statistics using local function
[mean_val, std_val] = calculate_stats(y);
fprintf('   Data statistics: Mean = %.4f, Std = %.4f\n', mean_val, std_val);

% Process data using local function
processed_data = process_signal(y, 0.05);
fprintf('   Signal processed with threshold = 0.05\n');

% Demonstrate function handles
fprintf('4. Demonstrating function handles...\n');
operation_handle = @(a, b) a .* b + sin(a);
result = apply_operation(x, y, operation_handle);
fprintf('   Applied custom operation via function handle\n');

% Display results
fprintf('5. Displaying results...\n');
figure('Name', 'Main Script Results');
subplot(2,2,1);
plot(x, y, 'b-', 'LineWidth', 1.5);
title('Original Signal');
xlabel('x'); ylabel('y');
grid on;

subplot(2,2,2);
plot(x, processed_data, 'r-', 'LineWidth', 1.5);
title('Processed Signal');
xlabel('x'); ylabel('Processed y');
grid on;

subplot(2,2,3);
plot(x, result, 'g-', 'LineWidth', 1.5);
title('Operation Result');
xlabel('x'); ylabel('Result');
grid on;

subplot(2,2,4);
histogram(y, 20, 'FaceColor', 'cyan', 'EdgeColor', 'black');
title('Data Distribution');
xlabel('Value'); ylabel('Frequency');
grid on;

fprintf('\nScript execution completed successfully!\n');
fprintf('Check the generated plots and results.\n');

% ========== LOCAL FUNCTIONS ==========

function [mean_val, std_val] = calculate_stats(data)
    % Calculate basic statistics for input data
    % Input: data - numerical array
    % Output: mean_val - mean of data, std_val - standard deviation
    
    mean_val = mean(data);
    std_val = std(data);
end

function processed = process_signal(signal, threshold)
    % Process signal by applying threshold-based filtering
    % Input: signal - input signal array
    %        threshold - filtering threshold
    % Output: processed - filtered signal
    
    % Simple threshold-based processing
    processed = signal;
    processed(abs(signal) < threshold) = 0;
    
    % Apply smoothing
    if length(signal) > 3
        processed = smooth(processed, 3);
    end
end

function result = apply_operation(x, y, func_handle)
    % Apply operation using function handle
    % Input: x, y - input arrays
    %        func_handle - function handle for operation
    % Output: result - result of operation
    
    result = func_handle(x, y);
end