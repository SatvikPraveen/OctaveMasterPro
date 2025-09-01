# Parallel Image Batch Processing

**Location:** `mini_projects/parallel_image_batch_processing/README.md`

## Overview

High-performance parallel image batch processing system demonstrating multicore processing, performance optimization, and scalable image processing workflows using Octave's parallel computing capabilities.

## Features

- Batch processing of large image datasets
- Parallel execution using multiple CPU cores
- Performance benchmarking and optimization
- Memory-efficient processing for large datasets
- Real-time progress monitoring and load balancing

## Files

- `batch_processor.m` - Main parallel processing controller
- `image_operations.m` - Core image processing functions
- `parallel_demo.m` - Demonstration and benchmarking script
- `performance_benchmark.m` - Performance testing and comparison

## Usage

```octave
# Run main demonstration
parallel_demo

# Batch process images
process_image_batch('input_images/', 'output_images/', @resize_operation, 'parallel', true);

# Performance benchmarking
benchmark_results = performance_benchmark('test_images/', {'resize', 'filter', 'enhance'});

# Custom parallel operation
results = parallel_image_operation(image_list, @custom_function, num_workers);
```

## Performance Features

- Automatic core detection and load balancing
- Memory usage optimization
- Progress tracking and ETA calculation
- Comparative analysis: sequential vs parallel

## Requirements

- Octave with parallel computing package
- Multiple CPU cores for parallel processing
- Sufficient RAM for batch processing

## Sample Outputs

- Processed image batches
- Performance benchmark reports
- Speedup analysis charts
- Resource utilization graphs
