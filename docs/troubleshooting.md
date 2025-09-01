# File location: OctaveMasterPro/docs/troubleshooting.md

# 🚨 Troubleshooting Guide

Comprehensive solutions for common issues in OctaveMasterPro.

## 🐳 Docker Issues

### Container Won't Start

**Problem**: `docker-compose up` fails with errors

**Solutions**:

```bash
# 1. Check Docker daemon
sudo systemctl status docker
sudo systemctl start docker

# 2. Clean build
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
docker-compose up

# 3. Check Docker resources
docker system df
docker system prune # If low space
```

### Port Already in Use

**Problem**: `Error: bind: address already in use`

**Solutions**:

```bash
# Find process using port 8888
lsof -i :8888
sudo kill -9 <PID>

# Use alternative port
docker-compose down
docker-compose up --scale octave-master=0
docker run -p 9999:8888 octave-master-pro_octave-master

# Or edit docker-compose.yml ports section
```

### Permission Denied Errors

**Problem**: Permission denied accessing files/directories

**Solutions**:

```bash
# Linux/macOS: Fix ownership
sudo chown -R $USER:$USER .
chmod +x *.sh
chmod -R 755 datasets/

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Windows: Run Docker Desktop as Administrator
```

### Memory Issues

**Problem**: Container crashes due to insufficient memory

**Solutions**:

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory → 6GB+

# Linux: Check available memory
free -h
# Close unnecessary applications

# Monitor container memory
docker stats octave_master_pro
```

## 📓 Jupyter Issues

### Kernel Not Found

**Problem**: Octave kernel missing from Jupyter

**Solutions**:

```bash
# Reinstall octave_kernel
pip3 uninstall octave_kernel
pip3 install octave_kernel
python3 -m octave_kernel install --user --force

# Verify installation
jupyter kernelspec list
# Should show 'octave' in list

# Restart Jupyter
docker-compose restart octave-master
```

### Notebook Won't Load

**Problem**: Jupyter notebook files won't open

**Solutions**:

```bash
# Check file permissions
ls -la notebooks/
chmod 644 notebooks/*.ipynb

# Clear Jupyter cache
rm -rf ~/.jupyter/runtime/
rm -rf ~/.local/share/jupyter/

# Check JSON syntax
python3 -m json.tool notebooks/00_environment_check.ipynb > /dev/null
# If error, file is corrupted
```

### Plots Not Displaying

**Problem**: Figures don't appear in notebooks

**Solutions**:

```octave
% Check graphics toolkit
graphics_toolkit('qt')

% Alternative toolkit
graphics_toolkit('gnuplot')

% Force inline plots (in notebook)
%plot inline

% Check display
figure();
plot(1:10);
% If still no plot, restart kernel
```

### Slow Performance

**Problem**: Jupyter runs slowly or freezes

**Solutions**:

```bash
# Increase resources
# Docker Desktop → Resources → CPUs → 4+

# Clear notebook outputs
jupyter nbconvert --clear-output notebooks/*.ipynb

# Restart with clean state
docker-compose down
docker volume rm octave-master-pro_jupyter-data
docker-compose up
```

## 🔧 Octave Issues

### Package Not Found

**Problem**: `error: package 'statistics' not found`

**Solutions**:

```octave
% Check installed packages
pkg list

% Install missing packages
pkg install -forge statistics
pkg install -forge image
pkg install -forge signal
pkg install -forge parallel

% Load packages
pkg load statistics
pkg load image
```

### Function Not Defined

**Problem**: `error: 'somefunction' undefined near line X`

**Solutions**:

```octave
% Check if function exists
which somefunction

% Check package availability
pkg list | grep function_name

% Load required package
pkg load package_name

% Check path
addpath('utils/');
addpath('notebooks/scripts_for_notebook_04/');
```

### Matrix Dimension Errors

**Problem**: `error: operator *: nonconformant arguments`

**Solutions**:

```octave
% Check dimensions
size(matrix_a)
size(matrix_b)

% Transpose if needed
result = matrix_a * matrix_b';

% Element-wise operations
result = matrix_a .* matrix_b;

% Reshape matrices
matrix_a = reshape(matrix_a, [m, n]);
```

### Memory Errors

**Problem**: `error: out of memory`

**Solutions**:

```octave
% Clear variables
clear all
pack

% Check memory usage
whos

% Process data in chunks
chunk_size = 1000;
for i = 1:chunk_size:total_size
    chunk_data = data(i:min(i+chunk_size-1, total_size), :);
    process_chunk(chunk_data);
end
```

## 📁 File System Issues

### Cannot Load Data Files

**Problem**: `error: load: unable to find file`

**Solutions**:

```octave
% Check current directory
pwd

% List files
ls

% Change to correct directory
cd datasets/

% Use full path
load('/workspace/OctaveMasterPro/datasets/sensor_data.mat');

% Check file exists
exist('sensor_data.mat', 'file')
```

### CSV Reading Errors

**Problem**: `readtable` fails or produces wrong results

**Solutions**:

```octave
% Check file format
type data1.csv | head

% Specify options
opts = detectImportOptions('data1.csv');
opts.Delimiter = ',';
opts.VariableNamesLine = 1;
data = readtable('data1.csv', opts);

% Alternative: Use csvread
data_matrix = csvread('data1.csv', 1, 0); % Skip header
```

### Path Issues

**Problem**: Scripts or utilities not found

**Solutions**:

```octave
% Check current path
path

% Add directories to path
addpath('utils/');
addpath('notebooks/scripts_for_notebook_04/');

% Save path permanently
savepath

% Use full paths in scripts
run('/workspace/OctaveMasterPro/utils/plot_utils.m');
```

## 🖥️ Platform-Specific Issues

### Windows Issues

**Problem**: Various Windows-specific errors

**Solutions**:

```batch
REM Use Windows Subsystem for Linux (WSL)
wsl --install

REM Or use PowerShell with admin privileges
REM Check Windows version compatibility

REM File path issues - use forward slashes
img = imread('datasets/images/samples/sample_01.jpg');
```

### macOS Issues

**Problem**: macOS permission or path issues

**Solutions**:

```bash
# Grant terminal full disk access
# System Preferences → Privacy → Full Disk Access

# Use Homebrew versions
brew install octave

# M1 Mac specific
arch -x86_64 brew install octave
```

### Linux Package Issues

**Problem**: Missing dependencies on Linux

**Solutions**:

```bash
# Update package lists
sudo apt update

# Install build essentials
sudo apt install build-essential

# Install graphics libraries
sudo apt install libqt5gui5 libqt5core5a

# For plotting issues
sudo apt install gnuplot-qt
```

## 📊 Data Issues

### Large File Loading

**Problem**: Files too large for memory

**Solutions**:

```octave
% Check available memory
memory

% Load in parts
data_part = readtable('large_file.csv', 'Range', 'A1:Z1000');

% Use memory mapping for large arrays
filename = 'large_data.dat';
m = memmapfile(filename, 'Format', 'double');
data = m.Data;
```

### Corrupt Data Files

**Problem**: Data files appear corrupted

**Solutions**:

```bash
# Regenerate datasets
cd datasets/
octave create_sensor_data.m
octave create_signal_analysis.m
octave create_sample_images.m

# Verify file integrity
file datasets/*.mat
head datasets/*.csv
```

## 🚀 Performance Issues

### Slow Execution

**Problem**: Code runs very slowly

**Solutions**:

```octave
% Profile code
profile on;
your_slow_function();
profile off;
profshow;

% Vectorize operations
% Bad:
for i = 1:length(x)
    y(i) = x(i)^2;
end

% Good:
y = x.^2;

% Preallocate arrays
n = 10000;
result = zeros(n, 1); % Preallocate
for i = 1:n
    result(i) = computation(i);
end
```

### Memory Leaks

**Problem**: Memory usage grows over time

**Solutions**:

```octave
% Clear variables regularly
clear temp_vars;

% Close figures
close all;

% Pack memory
pack;

% Monitor memory usage
while true
    % Your processing loop
    if mod(iteration, 100) == 0
        whos; % Check memory every 100 iterations
        pack; % Compress memory
    end
end
```

## 🔌 Integration Issues

### Git Integration

**Problem**: Git operations fail

**Solutions**:

```bash
# Initialize repository properly
git init
git add .
git commit -m "Initial commit"

# Handle large files with Git LFS
git lfs track "*.mat"
git lfs track "datasets/images/*"
git add .gitattributes

# Pre-commit setup
pip install pre-commit
pre-commit install
```

### IDE Integration

**Problem**: VSCode or other IDE issues

**Solutions**:

```bash
# Install extensions
# - Octave/MATLAB extension
# - Jupyter extension
# - Docker extension

# Configure workspace
# .vscode/settings.json:
{
    "octave.executable": "/usr/bin/octave",
    "files.associations": {
        "*.m": "octave"
    }
}
```

## 📞 Getting Help

### Log Collection

```bash
# Docker logs
docker-compose logs octave-master > debug.log

# Jupyter logs
docker-compose exec octave-master jupyter --paths

# System information
docker version
docker-compose version
octave --version
```

### Useful Commands

```bash
# Complete environment reset
docker-compose down -v
docker system prune -a
rm -rf .docker-volumes/
./init_project.sh
docker-compose up --build

# Health check
docker-compose exec octave-master octave --eval "pkg list"
curl -f http://localhost:8888/api || echo "Jupyter not responding"
```

### When to Seek Help

1. **Collect error messages** - exact text, line numbers
2. **Document steps** - what you tried, what failed
3. **System info** - OS version, Docker version, hardware specs
4. **Minimal example** - smallest code that reproduces issue
5. **Screenshots** - especially for GUI issues

### Support Channels

- **GitHub Issues**: Most technical problems
- **Stack Overflow**: General Octave/Jupyter questions
- **Docker Forums**: Container-specific issues
- **Documentation**: Check all markdown files first

**Remember**: Most issues have been solved before - search existing solutions first!
