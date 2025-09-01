# File location: OctaveMasterPro/docs/usage_examples.md

# 📚 Usage Examples

Common patterns and code snippets for OctaveMasterPro learning modules.

## 🏁 Quick Start Examples

### Loading and Inspecting Data

```octave
% Load CSV data
data = readtable('datasets/data1.csv');
head(data)

% Load MAT file
load('datasets/sensor_data.mat');
whos

% Basic statistics
mean(data.Salary)
std(data.Performance_Score)
```

### Basic Plotting

```octave
% Simple line plot
x = linspace(0, 2*pi, 100);
y = sin(x);
plot(x, y);
title('Sine Wave');
xlabel('x');
ylabel('sin(x)');

% Multiple plots
subplot(2,1,1);
plot(x, sin(x));
title('Sine');

subplot(2,1,2);
plot(x, cos(x));
title('Cosine');
```

## 📊 Data Analysis Patterns

### Working with Employee Data (data1.csv)

```octave
% Load data
data = readtable('datasets/data1.csv');

% Group by department
dept_stats = grpstats(data, 'Department', {'mean', 'std'}, 'DataVars', 'Salary');

% Filter active employees
active_data = data(strcmp(data.Active, 'Yes'), :);

% Age distribution
histogram(data.Age, 15);
title('Age Distribution');

% Salary vs Performance scatter
scatter(data.Performance_Score, data.Salary, 50, 'filled');
xlabel('Performance Score');
ylabel('Salary ($)');
title('Salary vs Performance');
```

### Time Series Analysis (stock_prices.csv)

```octave
% Load stock data
stocks = readtable('datasets/stock_prices.csv');

% Filter for specific stock
aapl = stocks(strcmp(stocks.Symbol, 'AAPL'), :);

% Convert dates
dates = datenum(aapl.Date);

% Price plot
plot(dates, aapl.Close);
datetick('x', 'yyyy-mm');
title('AAPL Stock Price');
ylabel('Price ($)');

% Calculate returns
returns = diff(log(aapl.Close));
histogram(returns, 50);
title('AAPL Daily Returns Distribution');

% Moving average
window = 20;
ma20 = movmean(aapl.Close, window);
plot(dates, aapl.Close, 'b-', dates, ma20, 'r-', 'LineWidth', 2);
legend('Price', '20-day MA');
```

## 🔬 Scientific Data Analysis

### Sensor Data Processing

```octave
% Load multi-dimensional sensor data
load('datasets/sensor_data.mat');

% Analyze temperature patterns
avg_temp_by_hour = squeeze(mean(mean(temperature_matrix, 1), 3));
plot(0:23, avg_temp_by_hour);
xlabel('Hour of Day');
ylabel('Temperature (°C)');
title('Average Temperature Profile');

% Pressure signal analysis
fs = 1000; % Sampling rate
t = (0:length(pressure_readings)-1) / fs;

plot(t(1:1000), pressure_readings(1:1000));
xlabel('Time (s)');
ylabel('Pressure (Pa)');
title('Pressure Signal (First Second)');

% FFT analysis
Y = fft(pressure_readings);
f = (0:length(Y)-1) * fs / length(Y);
plot(f(1:end/2), abs(Y(1:end/2)));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Pressure Signal Spectrum');
```

## 📈 Statistical Analysis Examples

### Experiment Data Analysis

```octave
% Load experiment data
exp_data = readtable('datasets/experiment_data.csv');

% Group comparison
control = exp_data(strcmp(exp_data.Group, 'Control'), :);
treat_a = exp_data(strcmp(exp_data.Group, 'Treatment_A'), :);
treat_b = exp_data(strcmp(exp_data.Group, 'Treatment_B'), :);

% Calculate improvement
control.Improvement = control.Post_Test - control.Pre_Test;
treat_a.Improvement = treat_a.Post_Test - treat_a.Pre_Test;
treat_b.Improvement = treat_b.Post_Test - treat_b.Pre_Test;

% Box plot comparison
boxplot([control.Improvement; treat_a.Improvement; treat_b.Improvement], ...
        [ones(height(control),1); 2*ones(height(treat_a),1); 3*ones(height(treat_b),1)]);
set(gca, 'XTickLabel', {'Control', 'Treatment A', 'Treatment B'});
ylabel('Test Score Improvement');
title('Treatment Effectiveness');

% T-test
[h, p] = ttest2(treat_b.Improvement, control.Improvement);
fprintf('Treatment B vs Control: p = %.4f\n', p);
```

## 🎛️ Signal Processing Examples

### Audio Signal Analysis

```octave
% Read audio file
[audio, fs] = audioread('datasets/signals/audio/sine_wave_440hz.wav');

% Time domain plot
t = (0:length(audio)-1) / fs;
plot(t, audio);
xlabel('Time (s)');
ylabel('Amplitude');
title('Audio Signal');

% Frequency analysis
Y = fft(audio);
f = (0:length(Y)-1) * fs / length(Y);
semilogx(f(1:end/2), 20*log10(abs(Y(1:end/2))));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Audio Spectrum');

% Spectrogram
spectrogram(audio, 1024, 512, 1024, fs, 'yaxis');
title('Audio Spectrogram');
```

### Filter Design and Application

```octave
% Load signal data
load('datasets/signal_analysis.mat');

% Design lowpass filter
fc = 100; % Cutoff frequency
[b, a] = butter(4, fc/(fs/2));

% Apply filter
filtered = filter(b, a, noisy_signal);

% Compare signals
subplot(3,1,1);
plot(t, clean_signal);
title('Clean Signal');

subplot(3,1,2);
plot(t, noisy_signal);
title('Noisy Signal');

subplot(3,1,3);
plot(t, filtered);
title('Filtered Signal');
```

## 🖼️ Image Processing Examples

### Basic Image Operations

```octave
% Load image
img = imread('datasets/images/samples/sample_01.jpg');
imshow(img);
title('Original Image');

% Convert to grayscale
gray_img = rgb2gray(img);
figure;
imshow(gray_img);
title('Grayscale');

% Edge detection
edges = edge(gray_img, 'canny');
figure;
imshow(edges);
title('Edge Detection');

% Histogram analysis
figure;
subplot(2,2,1); imhist(img(:,:,1)); title('Red Channel');
subplot(2,2,2); imhist(img(:,:,2)); title('Green Channel');
subplot(2,2,3); imhist(img(:,:,3)); title('Blue Channel');
subplot(2,2,4); imhist(gray_img); title('Grayscale');
```

### Batch Image Processing

```octave
% Process multiple images
image_dir = 'datasets/images/batch/';
output_dir = 'processed_images/';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Get list of images
image_files = dir([image_dir '*.jpg']);

for i = 1:length(image_files)
    % Load image
    img = imread([image_dir image_files(i).name]);

    % Apply processing (example: edge enhancement)
    gray = rgb2gray(img);
    edges = edge(gray, 'sobel');

    % Save result
    output_name = [output_dir 'processed_' image_files(i).name];
    imwrite(edges, output_name);

    fprintf('Processed %s\n', image_files(i).name);
end
```

## 🔄 Parallel Processing Examples

### Parallel Loops

```octave
% Check parallel capability
if exist('parfor')
    fprintf('Parallel processing available\n');
else
    fprintf('Sequential processing only\n');
end

% Parallel computation example
n = 1000;
results = zeros(1, n);

% Sequential version
tic;
for i = 1:n
    results(i) = expensive_calculation(i);
end
sequential_time = toc;

% Parallel version (if available)
tic;
parfor i = 1:n
    results(i) = expensive_calculation(i);
end
parallel_time = toc;

fprintf('Sequential: %.2f seconds\n', sequential_time);
fprintf('Parallel: %.2f seconds\n', parallel_time);
fprintf('Speedup: %.2fx\n', sequential_time / parallel_time);

function result = expensive_calculation(x)
    % Simulate computationally intensive task
    result = sum(sin(1:x*100));
end
```

## 📐 Linear Algebra Examples

### Matrix Operations

```octave
% Create matrices
A = randn(100, 100);
B = randn(100, 100);

% Basic operations
C = A * B;           % Matrix multiplication
D = A + B;           % Element-wise addition
E = A .* B;          % Element-wise multiplication

% Decompositions
[U, S, V] = svd(A);  % Singular Value Decomposition
[Q, R] = qr(A);      % QR Decomposition
[L, U, P] = lu(A);   % LU Decomposition

% Eigenvalues
[eigvec, eigval] = eig(A);

% Solve linear system
x = A \ randn(100, 1);
```

### Optimization Examples

```octave
% Function minimization
function y = rosenbrock(x)
    y = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
end

% Find minimum
x0 = [-1, 1];
[x_opt, fval] = fminunc(@rosenbrock, x0);

fprintf('Optimum: x = [%.4f, %.4f]\n', x_opt(1), x_opt(2));
fprintf('Function value: %.6f\n', fval);
```

## 🎯 Project Workflow Examples

### Mini Project Structure

```octave
% Mini project template
% 1. Load data
data = load_project_data();

% 2. Preprocessing
clean_data = preprocess_data(data);

% 3. Analysis
results = analyze_data(clean_data);

% 4. Visualization
create_visualizations(results);

% 5. Export results
save_results(results, 'project_output.mat');
```

### Flagship Project Pipeline

```octave
% Complete data science pipeline
function run_flagship_project()
    % Data ingestion
    raw_data = load_multiple_datasets();

    % Data cleaning and validation
    clean_data = data_cleaning_pipeline(raw_data);

    % Feature engineering
    features = extract_features(clean_data);

    % Statistical modeling
    model = build_predictive_model(features);

    % Model validation
    performance = validate_model(model, features);

    % Visualization dashboard
    create_dashboard(model, performance);

    % Generate report
    generate_report(model, performance, 'flagship_report.pdf');
end
```

## 🔍 Debugging Tips

### Common Issues

```octave
% Check variable types
class(my_variable)
size(my_variable)

% Memory usage
whos

% Clear variables
clear variable_name
clear all  % Clear everything

% Check loaded packages
pkg list

% Debugging plots
figure; plot(debug_data); title('Debug Plot');
```

### Performance Profiling

```octave
% Profile code execution
profile on;
your_function(inputs);
profile off;
profshow;
```

## 📝 Best Practices

### Code Organization

- Use descriptive variable names
- Add comments for complex operations
- Break large scripts into functions
- Use consistent indentation (2 spaces)

### Data Handling

- Always check data dimensions with `size()`
- Validate data types before processing
- Handle missing data appropriately
- Use vectorized operations when possible

### Visualization

- Always label axes and add titles
- Use appropriate plot types for data
- Consider colorblind-friendly palettes
- Save plots in appropriate formats

### Performance

- Vectorize operations instead of loops
- Preallocate arrays when possible
- Use built-in functions over custom implementations
- Profile code to identify bottlenecks

These examples provide practical starting points for all major learning modules and project types in OctaveMasterPro!
