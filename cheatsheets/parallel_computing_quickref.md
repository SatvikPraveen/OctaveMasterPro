# Parallel Computing Quick Reference

## Parallel Package Setup

### Installation and Loading

```octave
pkg install -forge parallel     % Install parallel package
pkg load parallel               % Load parallel package
pkg list                        % Check installed packages
```

### Capability Check

```octave
% Check if parallel processing is available
if exist('parfor', 'builtin')
    fprintf('Parallel processing available\n');
else
    fprintf('Parallel processing not available\n');
end
```

## Core Parallel Constructs

### Parallel For Loops (parfor)

```octave
% Basic parallel for loop
n = 1000;
result = zeros(1, n);

parfor i = 1:n
    result(i) = expensive_computation(i);
end

% With reduction variable
total = 0;
parfor i = 1:n
    total = total + data(i);    % Automatic reduction
end
```

### Parallel Function Application

```octave
% Apply function to array elements in parallel
data = 1:1000;
result = arrayfun(@(x) x^2, data);  % May use parallel processing

% Explicit parallel apply
cellfun(@process_func, data_cell, 'UniformOutput', false);
```

## System Resources

### Worker Management

```octave
% Get number of CPU cores
n_cores = nproc();              % Get processor count

% Optimal worker count (typically cores - 1)
n_workers = max(1, n_cores - 1);

% Check memory constraints
memory_info = memory();
available_gb = memory_info.MemAvailableAllArrays / 1e9;
```

### Performance Monitoring

```octave
% Time parallel vs serial execution
tic;
serial_result = process_data_serial(data);
serial_time = toc;

tic;
parallel_result = process_data_parallel(data);
parallel_time = toc;

speedup = serial_time / parallel_time;
fprintf('Speedup: %.2fx\n', speedup);
```

## Common Parallel Patterns

### Embarrassingly Parallel Problems

```octave
% Monte Carlo simulation
n_simulations = 10000;
results = zeros(1, n_simulations);

parfor i = 1:n_simulations
    results(i) = monte_carlo_trial();
end

final_estimate = mean(results);
```

### Data Parallel Operations

```octave
% Element-wise operations (automatically parallel)
A = rand(1000, 1000);
B = rand(1000, 1000);
C = A .* B;                     % Parallel element-wise multiplication

% Matrix operations
result = A * B;                 % May use parallel BLAS
eigenvals = eig(A);             % Parallel eigenvalue computation
```

### Image Processing

```octave
% Process multiple images in parallel
image_files = dir('*.jpg');
processed_images = cell(length(image_files), 1);

parfor i = 1:length(image_files)
    img = imread(image_files(i).name);
    processed_images{i} = process_image(img);
end
```

## Chunking Strategies

### Manual Chunking

```octave
data = 1:10000;
chunk_size = 1000;
n_chunks = ceil(length(data) / chunk_size);
results = cell(n_chunks, 1);

parfor chunk = 1:n_chunks
    start_idx = (chunk - 1) * chunk_size + 1;
    end_idx = min(chunk * chunk_size, length(data));
    chunk_data = data(start_idx:end_idx);
    results{chunk} = process_chunk(chunk_data);
end

% Combine results
final_result = cell2mat(results);
```

### Automatic Load Balancing

```octave
% Let Octave handle load balancing
data = generate_irregular_workload();

parfor i = 1:length(data)
    results(i) = variable_time_computation(data(i));
end
```

## Memory Management

### Memory-Efficient Parallel Processing

```octave
% Process large datasets in chunks
large_data = load_big_dataset();
max_memory_gb = 2.0;
element_size = 8;  % bytes per double

max_elements = (max_memory_gb * 1e9) / element_size;
chunk_size = floor(max_elements / n_workers);

n_chunks = ceil(size(large_data, 1) / chunk_size);

parfor chunk = 1:n_chunks
    start_row = (chunk - 1) * chunk_size + 1;
    end_row = min(chunk * chunk_size, size(large_data, 1));

    chunk_data = large_data(start_row:end_row, :);
    chunk_results{chunk} = process_large_chunk(chunk_data);

    clear chunk_data;  % Free memory immediately
end
```

### Reducing Memory Overhead

```octave
% Use broadcast variables for read-only data
shared_parameters = load_parameters();

parfor i = 1:n_tasks
    % shared_parameters is automatically broadcast to workers
    results(i) = compute_with_params(task_data(i), shared_parameters);
end
```

## Parallel Algorithms

### Parallel Reduction

```octave
% Sum reduction
data = rand(1, 10000);
partial_sums = zeros(1, n_workers);

parfor worker = 1:n_workers
    start_idx = floor((worker-1) * length(data) / n_workers) + 1;
    end_idx = floor(worker * length(data) / n_workers);
    partial_sums(worker) = sum(data(start_idx:end_idx));
end

total_sum = sum(partial_sums);
```

### Parallel Search

```octave
% Find maximum element in parallel
data = rand(1, 100000);
n_workers = 4;
worker_maxes = zeros(1, n_workers);
worker_indices = zeros(1, n_workers);

parfor worker = 1:n_workers
    start_idx = floor((worker-1) * length(data) / n_workers) + 1;
    end_idx = floor(worker * length(data) / n_workers);

    [worker_maxes(worker), local_idx] = max(data(start_idx:end_idx));
    worker_indices(worker) = start_idx + local_idx - 1;
end

[global_max, winner] = max(worker_maxes);
global_index = worker_indices(winner);
```

## Signal Processing Parallelism

### Parallel FFT

```octave
% Process multiple signals in parallel
signals = cell(1, n_signals);
fft_results = cell(1, n_signals);

parfor i = 1:n_signals
    fft_results{i} = fft(signals{i});
end
```

### Filter Bank Processing

```octave
% Apply different filters in parallel
signal = randn(1, 10000);
filters = {low_pass_filter, high_pass_filter, band_pass_filter};
filtered_signals = cell(length(filters), 1);

parfor i = 1:length(filters)
    filtered_signals{i} = filter(filters{i}.b, filters{i}.a, signal);
end
```

## Statistical Computing

### Bootstrap Sampling

```octave
data = randn(100, 1);
n_bootstrap = 10000;
bootstrap_means = zeros(1, n_bootstrap);

parfor i = 1:n_bootstrap
    bootstrap_sample = datasample(data, length(data), 'Replace', true);
    bootstrap_means(i) = mean(bootstrap_sample);
end

confidence_interval = quantile(bootstrap_means, [0.025, 0.975]);
```

### Cross-Validation

```octave
% K-fold cross-validation
k_folds = 10;
cv_errors = zeros(1, k_folds);
fold_size = floor(size(data, 1) / k_folds);

parfor fold = 1:k_folds
    test_start = (fold - 1) * fold_size + 1;
    test_end = min(fold * fold_size, size(data, 1));

    test_indices = test_start:test_end;
    train_indices = setdiff(1:size(data, 1), test_indices);

    model = train_model(data(train_indices, :));
    predictions = predict_model(model, data(test_indices, :));
    cv_errors(fold) = compute_error(predictions, labels(test_indices));
end

mean_cv_error = mean(cv_errors);
```

## Optimization Techniques

### Vectorization First

```octave
% Prefer vectorized operations over parallel loops
% Good: Vectorized (automatically parallel)
result = A .* B + C;

% Less optimal: Explicit parallel loop
parfor i = 1:numel(A)
    result(i) = A(i) * B(i) + C(i);
end
```

### Minimize Communication

```octave
% Bad: Frequent communication
global_sum = 0;
parfor i = 1:n
    global_sum = global_sum + compute(i);  % Communication overhead
end

% Good: Local accumulation, single reduction
local_sums = zeros(1, n_workers);
parfor worker = 1:n_workers
    local_sum = 0;
    start_idx = worker_chunk_start(worker);
    end_idx = worker_chunk_end(worker);

    for i = start_idx:end_idx
        local_sum = local_sum + compute(i);
    end
    local_sums(worker) = local_sum;
end
global_sum = sum(local_sums);
```

### Load Balancing

```octave
% For irregular workloads, use dynamic scheduling
work_items = generate_variable_work();

% Octave handles dynamic load balancing automatically
parfor i = 1:length(work_items)
    results(i) = process_work_item(work_items(i));
end
```

## Debugging Parallel Code

### Common Issues

```octave
% Variable classification issues
% Make sure loop variables are properly classified

% Transparency violations
% Avoid unclear variable dependencies

% Check for race conditions
% Ensure no shared mutable state
```

### Debugging Tips

```octave
% Use fprintf for debugging (thread-safe)
parfor i = 1:n
    fprintf('Processing item %d\n', i);
    result(i) = process_item(i);
end

% Start with small problem sizes
% Test serial version first
% Gradually increase parallelism
```

## Performance Guidelines

### When to Use Parallel Processing

- **Use for**: CPU-intensive tasks, independent iterations, large datasets
- **Avoid for**: I/O heavy tasks, small problem sizes, highly dependent calculations

### Parallel Overhead Considerations

```octave
% Minimum work per iteration guideline
min_work_time = 0.001;  % 1ms minimum

% Check if parallel processing is worth it
if estimated_serial_time < (n_workers * min_work_time)
    % Use serial processing
    for i = 1:n
        results(i) = quick_computation(i);
    end
else
    % Use parallel processing
    parfor i = 1:n
        results(i) = expensive_computation(i);
    end
end
```

## Best Practices Checklist

- ✅ **Profile first**: Identify bottlenecks before parallelizing
- ✅ **Start simple**: Begin with parfor on independent loops
- ✅ **Minimize communication**: Reduce data transfer between workers
- ✅ **Chunk appropriately**: Balance load vs overhead
- ✅ **Handle memory**: Monitor memory usage with multiple workers
- ✅ **Test scaling**: Verify speedup with different worker counts
- ✅ **Use vectorization**: Leverage built-in parallel operations
- ✅ **Plan for serial fallback**: Handle cases where parallel isn't available

## Quick Reference Commands

| Operation      | Syntax                       | Description               |
| -------------- | ---------------------------- | ------------------------- |
| Parallel loop  | `parfor i = 1:n`             | Parallel for loop         |
| Worker count   | `nproc()`                    | Get CPU core count        |
| Memory info    | `memory()`                   | System memory information |
| Time execution | `tic; code; toc`             | Measure execution time    |
| Check parallel | `exist('parfor', 'builtin')` | Test availability         |
| Load package   | `pkg load parallel`          | Load parallel package     |

## Troubleshooting

### Performance Issues

- Check for load imbalance
- Reduce communication overhead
- Increase chunk sizes
- Verify adequate work per iteration

### Memory Problems

- Monitor memory usage per worker
- Implement chunking for large data
- Clear variables explicitly
- Use memory-efficient algorithms
