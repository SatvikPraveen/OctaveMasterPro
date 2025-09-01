#!/bin/bash

# OctaveMasterPro Project Structure Initialization Script
# Creates the complete directory structure and placeholder files
# Usage: chmod +x init_project.sh && ./init_project.sh

set -e  # Exit on any error

PROJECT_NAME="OctaveMasterPro"
CURRENT_DIR=$(pwd)

echo "🚀 Initializing $PROJECT_NAME project structure..."
echo "📍 Creating project in: $CURRENT_DIR/$PROJECT_NAME"

# Create main project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

echo "📁 Creating directory structure..."

# Create main directories
mkdir -p notebooks
mkdir -p notebooks/scripts_for_notebook_04
mkdir -p cheatsheets
mkdir -p mini_projects/{signal_processing_simulation,image_processing_basics,stock_market_analysis,parallel_image_batch_processing}
mkdir -p flagship_project/{project_scripts,datasets,report}
mkdir -p utils
mkdir -p datasets/{images,signals}
mkdir -p docs

echo "📝 Creating notebook files..."

# Create notebook files
cat > notebooks/00_environment_check.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Environment Check\n",
    "Verify Octave setup, installed packages, and plotting capabilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "% Check Octave version\n",
    "version\n",
    "\n",
    "% Check installed packages\n",
    "pkg list\n",
    "\n",
    "% Test basic plotting\n",
    "x = linspace(0, 2*pi, 100);\n",
    "y = sin(x);\n",
    "plot(x, y);\n",
    "title('Environment Test Plot');\n",
    "xlabel('x');\n",
    "ylabel('sin(x)');"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Octave",
   "language": "octave",
   "name": "octave"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create other notebook placeholders
notebooks=(
    "01_basics_intro.ipynb:Variables, data types, operators, I/O"
    "02_vectors_matrices.ipynb:Vector/matrix creation, manipulation"
    "03_indexing_logic.ipynb:Indexing, logical ops, boolean masks"
    "04_scripts_functions.ipynb:Scripts, functions, scopes, handles"
    "05_data_handling_files.ipynb:File I/O, CSV, MAT, large datasets"
    "06_plotting_2d_3d.ipynb:2D/3D visualization, customization"
    "07_linear_algebra.ipynb:Matrix decompositions, eigensystems"
    "08_statistics_analysis.ipynb:Stats, hypothesis testing, regression"
    "09_optimization_roots.ipynb:Optimization, root-finding techniques"
    "10_advanced_programming.ipynb:OOP, GUIs, event-driven programming"
    "11_expert_topics.ipynb:Performance tuning, hybrid pipelines"
    "12_parallel_computing.ipynb:Multicore tasks, parfor-equivalents"
)

for notebook_info in "${notebooks[@]}"; do
    IFS=':' read -r notebook_name notebook_desc <<< "$notebook_info"
    cat > "notebooks/$notebook_name" << EOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ${notebook_name%.*}\n",
    "$notebook_desc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "% Content to be added"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Octave",
   "language": "octave",
   "name": "octave"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
done

echo "📋 Creating script files..."

# Create scripts for notebook 04
cat > notebooks/scripts_for_notebook_04/main_script.m << 'EOF'
% Main script demonstrating script execution and function calls
% Used in notebook 04_scripts_functions.ipynb

fprintf('Running main_script.m\n');

% Call another script
script_zero;

% Define and call a local function
result = calculate_sum(5, 10);
fprintf('Sum result: %d\n', result);

function sum_val = calculate_sum(a, b)
    sum_val = a + b;
end
EOF

cat > notebooks/scripts_for_notebook_04/script_zero.m << 'EOF'
% Secondary script called by main_script.m
% Demonstrates script composition

fprintf('Executing script_zero.m\n');
x = linspace(0, pi, 50);
y = cos(x);
plot(x, y);
title('Script Zero Plot');
EOF

echo "📚 Creating cheatsheets..."

# Create cheatsheet files
cat > cheatsheets/octave_syntax_cheatsheet.md << 'EOF'
# Octave Syntax Cheatsheet

## Basic Operations
```octave
% Variables and assignment
x = 5;
y = [1, 2, 3];
z = [1; 2; 3];

% Basic math
a + b, a - b, a * b, a / b, a ^ b

% Element-wise operations  
a .* b, a ./ b, a .^ b
```

## Matrix Operations
```octave
% Create matrices
A = [1, 2; 3, 4];
B = zeros(2, 3);
C = ones(3, 2);
D = eye(3);

% Matrix operations
A * B     % Matrix multiplication
A'        % Transpose
inv(A)    % Inverse
det(A)    % Determinant
```

## Control Flow
```octave
% If statement
if condition
    % code
elseif other_condition
    % code
else
    % code
end

% For loop
for i = 1:10
    % code
end

% While loop
while condition
    % code
end
```
EOF

cat > cheatsheets/plotting_cheatsheet.md << 'EOF'
# Plotting Cheatsheet

## 2D Plots
```octave
% Basic plot
plot(x, y);
plot(x, y, 'r-', 'LineWidth', 2);

% Multiple plots
hold on;
plot(x, y1, 'b-');
plot(x, y2, 'r--');
hold off;

% Subplots
subplot(2, 1, 1);
plot(x, y1);
subplot(2, 1, 2);
plot(x, y2);
```

## 3D Plots
```octave
% Surface plot
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2);
Z = X.^2 + Y.^2;
surf(X, Y, Z);

% 3D line plot
plot3(x, y, z);
```

## Customization
```octave
title('Plot Title');
xlabel('X Label');
ylabel('Y Label');
legend('Series 1', 'Series 2');
grid on;
axis([xmin, xmax, ymin, ymax]);
```
EOF

cat > cheatsheets/linear_algebra_quickref.md << 'EOF'
# Linear Algebra Quick Reference

## Matrix Decompositions
```octave
% LU decomposition
[L, U, P] = lu(A);

% QR decomposition
[Q, R] = qr(A);

% SVD
[U, S, V] = svd(A);

% Eigenvalues and eigenvectors
[V, D] = eig(A);
```

## Solving Systems
```octave
% Linear system Ax = b
x = A \ b;          % Preferred method
x = inv(A) * b;     % Less efficient

% Least squares
x = A \ b;          % Works for overdetermined systems
```

## Matrix Properties
```octave
rank(A)             % Matrix rank
cond(A)             % Condition number
norm(A)             % Matrix norm
trace(A)            % Trace
```
EOF

cat > cheatsheets/parallel_computing_quickref.md << 'EOF'
# Parallel Computing Quick Reference

## Basic Parallel Operations
```octave
% Load parallel package
pkg load parallel

% Check available cores
nproc()

% Parallel for loop (parfor in MATLAB)
parcellfun(@(x) expensive_function(x), data_cell, 'UniformOutput', false);

% Parallel map
result = pararrayfun(@sin, [1:1000]);
```

## Performance Tips
```octave
% Vectorize operations
y = sin(x);         % Instead of loops

% Preallocate arrays
result = zeros(1000, 1);

% Use built-in functions
sum(x)              % Instead of manual summation
```
EOF

echo "🛠️ Creating utility files..."

# Create utility files
cat > utils/plot_utils.m << 'EOF'
function setup_plot_defaults()
    % Setup default plotting parameters for consistent styling
    
    set(0, 'defaultaxesfontsize', 12);
    set(0, 'defaultaxesfontname', 'Arial');
    set(0, 'defaultlinelinewidth', 1.5);
    set(0, 'defaultaxesgrid', 'on');
end

function save_figure(filename, fig_handle)
    % Save figure with consistent settings
    if nargin < 2
        fig_handle = gcf;
    end
    
    print(fig_handle, filename, '-dpng', '-r300');
    fprintf('Figure saved as: %s\n', filename);
end
EOF

cat > utils/data_loader.m << 'EOF'
function data = load_csv_data(filename)
    % Load CSV data with error handling
    
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    
    try
        data = csvread(filename);
        fprintf('Loaded data: %dx%d\n', size(data, 1), size(data, 2));
    catch ME
        error('Error loading CSV: %s', ME.message);
    end
end

function data = load_mat_data(filename)
    % Load MAT file data
    
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    
    data = load(filename);
    fprintf('Loaded MAT file with variables: %s\n', strjoin(fieldnames(data), ', '));
end
EOF

cat > utils/parallel_wrappers.m << 'EOF'
function result = parallel_map(func, data, varargin)
    % Wrapper for parallel mapping operations
    
    pkg load parallel;
    
    if iscell(data)
        result = parcellfun(func, data, varargin{:});
    else
        result = pararrayfun(func, data, varargin{:});
    end
end

function benchmark_parallel(func, data, num_trials)
    % Benchmark serial vs parallel execution
    
    if nargin < 3
        num_trials = 5;
    end
    
    % Serial timing
    tic;
    for i = 1:num_trials
        result_serial = arrayfun(func, data);
    end
    time_serial = toc / num_trials;
    
    % Parallel timing
    pkg load parallel;
    tic;
    for i = 1:num_trials
        result_parallel = pararrayfun(func, data);
    end
    time_parallel = toc / num_trials;
    
    speedup = time_serial / time_parallel;
    fprintf('Serial time: %.4f s\n', time_serial);
    fprintf('Parallel time: %.4f s\n', time_parallel);
    fprintf('Speedup: %.2fx\n', speedup);
end
EOF

echo "📊 Creating sample datasets..."

# Create sample CSV data
cat > datasets/data1.csv << 'EOF'
x,y,z
1,2,3
4,5,6
7,8,9
10,11,12
EOF

echo "📋 Creating flagship project files..."

# Create flagship project notebook
cat > flagship_project/project_notebook.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Smart Environmental Sensor Data Analysis System\n",
    "## Flagship Project Demonstrating Advanced Octave Capabilities\n",
    "\n",
    "This project showcases:\n",
    "- IoT data ingestion and preprocessing\n",
    "- Statistical analysis and anomaly detection\n",
    "- Monte Carlo simulations\n",
    "- Parallel processing optimization\n",
    "- Interactive dashboards and reporting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "% Load required packages\n",
    "pkg load statistics\n",
    "pkg load parallel\n",
    "\n",
    "% Add project scripts to path\n",
    "addpath('project_scripts/');\n",
    "\n",
    "fprintf('Flagship project environment ready\\n');"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Octave",
   "language": "octave",
   "name": "octave"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

cat > flagship_project/parallelized_pipeline_demo.m << 'EOF'
% Parallelized Pipeline Demo
% Demonstrates high-performance data processing

function parallelized_pipeline_demo()
    fprintf('Starting parallelized pipeline demo...\n');
    
    % Generate sample sensor data
    data = generate_sensor_data(10000);
    
    % Process data in parallel
    pkg load parallel;
    processed_data = parallel_process_sensors(data);
    
    % Create visualization
    visualize_results(processed_data);
    
    fprintf('Pipeline demo complete!\n');
end

function data = generate_sensor_data(n_samples)
    % Generate synthetic sensor data
    t = linspace(0, 24, n_samples); % 24 hour period
    
    % Temperature with daily cycle + noise
    temp = 20 + 10*sin(2*pi*t/24) + randn(size(t));
    
    % Humidity inversely correlated with temperature
    humidity = 80 - 0.5*(temp - 20) + 5*randn(size(t));
    
    % Pressure with random walk
    pressure = 1013 + cumsum(randn(size(t))*0.1);
    
    data.time = t;
    data.temperature = temp;
    data.humidity = humidity;
    data.pressure = pressure;
end

function result = parallel_process_sensors(data)
    % Process sensor data using parallel operations
    
    sensors = {'temperature', 'humidity', 'pressure'};
    result = struct();
    
    for i = 1:length(sensors)
        sensor = sensors{i};
        raw_data = data.(sensor);
        
        % Parallel processing of each sensor stream
        processed = pararrayfun(@(x) smooth_and_filter(x), raw_data);
        result.(sensor) = processed;
    end
    
    result.time = data.time;
end

function filtered = smooth_and_filter(value)
    % Example processing function
    filtered = value * 0.9 + 0.1 * randn(); % Simple filter with noise
end

function visualize_results(data)
    % Create visualization of processed results
    
    figure;
    subplot(3, 1, 1);
    plot(data.time, data.temperature);
    title('Processed Temperature Data');
    ylabel('Temperature (°C)');
    
    subplot(3, 1, 2);
    plot(data.time, data.humidity);
    title('Processed Humidity Data');
    ylabel('Humidity (%)');
    
    subplot(3, 1, 3);
    plot(data.time, data.pressure);
    title('Processed Pressure Data');
    ylabel('Pressure (hPa)');
    xlabel('Time (hours)');
end
EOF

echo "🐳 Creating Docker configuration..."

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    octave \
    octave-parallel \
    octave-signal \
    octave-statistics \
    octave-image \
    octave-io \
    octave-control \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    gnuplot-x11 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for Jupyter
RUN pip3 install jupyter jupyterlab octave_kernel

# Set up Octave kernel for Jupyter
RUN python3 -m octave_kernel install --user

# Create working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  octave-jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - .:/workspace
    environment:
      - JUPYTER_ENABLE_LAB=yes
    command: >
      bash -c "
        echo 'Starting OctaveMasterPro environment...' &&
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
      "
EOF

# Create .dockerignore
cat > .dockerignore << 'EOF'
.git
.gitignore
README.md
*.log
*.tmp
.DS_Store
Thumbs.db
EOF

echo "📖 Creating documentation..."

# Create README.md
cat > README.md << 'EOF'
# 🎯 OctaveMasterPro

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Octave](https://img.shields.io/badge/Octave-6.x-orange)](https://octave.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Master MATLAB-compatible computing the modern way.**

A comprehensive learning and project environment for GNU Octave featuring Docker-based setup, 12 progressive learning modules, real-world projects, and parallel computing workflows.

## 🚀 Quick Start

```bash
# Clone and start
git clone <your-repo-url>
cd OctaveMasterPro
docker-compose up --build

# Open browser to http://localhost:8888
```

## 📚 What's Inside

- **12 Progressive Notebooks**: From basics to parallel computing
- **4 Mini Projects**: Signal processing, image analysis, stock market data
- **1 Flagship Project**: Smart environmental sensor analysis system
- **Ready-to-use Cheatsheets**: Quick reference for syntax and functions
- **Docker Environment**: Zero local setup required

## 🎯 Learning Path

1. Start with `00_environment_check.ipynb`
2. Progress through numbered notebooks
3. Practice with mini projects
4. Showcase skills with the flagship project

## 🛠️ Features

- ✅ MATLAB-compatible Octave programming
- ✅ Docker + Jupyter integration
- ✅ Parallel computing demonstrations
- ✅ Real-world datasets and projects
- ✅ Professional visualization examples
- ✅ Portfolio-ready outputs

## 📊 Project Structure

See the complete structure in `docs/architecture_diagram.svg`

## 🤝 Contributing

Contributions welcome! Please read our contribution guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.
EOF

# Create setup_instructions.md
cat > setup_instructions.md << 'EOF'
# Setup Instructions

## Prerequisites
- Docker and Docker Compose installed
- Git (for cloning)
- Web browser

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd OctaveMasterPro
   ```

2. **Build and start the environment**
   ```bash
   docker-compose up --build
   ```

3. **Access Jupyter Lab**
   - Open browser to: http://localhost:8888
   - No password required for local development

4. **Start learning**
   - Begin with `notebooks/00_environment_check.ipynb`
   - Follow the numbered sequence

## Troubleshooting

### Port 8888 already in use
```bash
# Use different port
docker-compose up --build -d
docker-compose exec octave-jupyter jupyter lab --port=8889
```

### Docker build issues
```bash
# Clean rebuild
docker-compose down
docker system prune -f
docker-compose up --build
```

### Octave packages not found
The Docker image includes all required packages. If issues persist:
```bash
# Inside Jupyter, run:
pkg list  # Check installed packages
pkg load parallel  # Load specific package
```

## Development Tips

- All notebooks auto-save
- Use `Ctrl+Enter` to run cells
- Restart kernel if memory issues occur
- Save important outputs before stopping container
EOF

# Create LICENSE
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 OctaveMasterPro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, such so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. cd $PROJECT_NAME"
echo "2. Review and customize the generated files"
echo "3. Initialize git repository: git init"
echo "4. Start the environment: docker-compose up --build"
echo "5. Open browser to: http://localhost:8888"
echo ""
echo "🎯 Your OctaveMasterPro project is ready!"
echo "📁 Location: $CURRENT_DIR/$PROJECT_NAME"