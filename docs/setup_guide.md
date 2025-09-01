# File location: OctaveMasterPro/docs/setup_guide.md

# 🚀 OctaveMasterPro Setup Guide

Complete setup instructions for all platforms and use cases.

## 📋 Prerequisites

### System Requirements

- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 10GB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Required Software

- **Docker Desktop** (recommended method)
- **Git** for version control
- **Text editor** (VSCode recommended)

## 🐳 Docker Setup (Recommended)

### Step 1: Install Docker

```bash
# Windows/macOS: Download Docker Desktop
# https://docs.docker.com/get-docker/

# Ubuntu:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/OctaveMasterPro.git
cd OctaveMasterPro
```

### Step 3: Initialize Project

```bash
chmod +x init_project.sh
./init_project.sh
```

### Step 4: Launch Environment

```bash
# Start Jupyter Lab
docker-compose up

# Alternative: Detached mode
docker-compose up -d

# Access CLI
docker-compose --profile cli run octave-cli
```

### Step 5: Access Jupyter

- **URL**: http://localhost:8888
- **Alternative**: http://localhost:8889
- **No password required** in development mode

## 💻 Local Installation

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Octave and packages
sudo apt install octave \
    octave-parallel \
    octave-statistics \
    octave-image \
    octave-signal \
    octave-control \
    octave-optim \
    octave-io

# Install Python and Jupyter
sudo apt install python3 python3-pip
pip3 install jupyter jupyterlab octave_kernel

# Install Octave kernel
python3 -m octave_kernel install --user

# Start Jupyter
cd OctaveMasterPro
jupyter lab
```

### macOS (Homebrew)

```bash
# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Octave
brew install octave

# Install Python packages
pip3 install jupyter jupyterlab octave_kernel

# Install kernel
python3 -m octave_kernel install --user

# Launch
cd OctaveMasterPro
jupyter lab
```

### Windows

```batch
REM Download and install Octave from octave.org
REM Download Python from python.org

REM Install Jupyter
pip install jupyter jupyterlab octave_kernel

REM Install kernel
python -m octave_kernel install --user

REM Launch
cd OctaveMasterPro
jupyter lab
```

## 🔧 Verification Steps

### 1. Test Docker Environment

```bash
# Check containers
docker-compose ps

# Check logs
docker-compose logs octave-master

# Test Octave
docker-compose exec octave-master octave --eval "disp('Hello Octave!')"
```

### 2. Test Jupyter Integration

```bash
# List kernels
jupyter kernelspec list

# Expected output should include 'octave'
```

### 3. Test Package Installation

Open Octave and verify packages:

```octave
pkg list
```

Required packages:

- statistics
- image
- signal
- parallel
- optim
- io
- control

### 4. Run Environment Check

1. Open `notebooks/00_environment_check.ipynb`
2. Run all cells (Ctrl+Shift+Enter)
3. Verify:
   - No errors in code execution
   - Plots display correctly
   - All packages load successfully
   - Performance benchmarks complete

## 🛠️ Advanced Configuration

### Custom Port Configuration

```yaml
# docker-compose.override.yml
version: "3.8"
services:
  octave-master:
    ports:
      - "9999:8888" # Custom port
```

### Memory Optimization

```bash
# Increase Docker memory (Docker Desktop)
# Settings → Resources → Memory → 6GB

# Linux: Edit daemon.json
sudo nano /etc/docker/daemon.json
{
  "default-runtime": "runc",
  "default-ulimits": {
    "memlock": {"hard": -1, "soft": -1}
  }
}
```

### Volume Optimization

```yaml
# Use named volumes for better performance
volumes:
  octave_cache:
    driver: local
```

### Performance Tuning

```octave
% Octave configuration (~/.octaverc)
graphics_toolkit('qt');
set(0, 'defaultfigurerenderer', 'opengl');

% Memory optimization
clear all
pack
```

## 🌐 Network Configuration

### Proxy Settings

```bash
# Docker with corporate proxy
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

docker-compose up --build
```

### Firewall Configuration

Allow these ports:

- **8888**: Jupyter Lab
- **8889**: Alternative Jupyter port

```bash
# Ubuntu UFW
sudo ufw allow 8888
sudo ufw allow 8889

# Windows Firewall
# Use Windows Defender Firewall → Allow apps
```

## 📊 Development Workflow

### Daily Usage

```bash
# 1. Start environment
docker-compose up -d

# 2. Open Jupyter
# http://localhost:8888

# 3. Work on notebooks
# Start with 00_environment_check.ipynb

# 4. Stop environment
docker-compose down
```

### Project Development

```bash
# Create new branch
git checkout -b feature/new-analysis

# Work on code
# Commit frequently

# Push changes
git add .
git commit -m "Add new analysis module"
git push origin feature/new-analysis
```

### Code Quality

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run checks
pre-commit run --all-files
```

## 🚨 Quick Fixes

### Port Already in Use

```bash
# Find process using port
lsof -i :8888
sudo kill -9 <PID>

# Or use different port
docker-compose up --scale octave-master=0
docker run -p 9999:8888 octave-master-pro_octave-master
```

### Permission Denied

```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x *.sh

# Docker permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Container Won't Start

```bash
# Complete reset
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
docker-compose up
```

### Jupyter Kernel Issues

```bash
# Reinstall kernel
pip3 uninstall octave_kernel
pip3 install octave_kernel
python3 -m octave_kernel install --user --force

# Clear Jupyter cache
jupyter --paths
rm -rf ~/.jupyter/kernels/octave
```

## ✅ Success Checklist

- [ ] Docker containers start without errors
- [ ] Jupyter Lab accessible at localhost:8888
- [ ] Octave kernel appears in kernel list
- [ ] Environment check notebook runs completely
- [ ] All plots display correctly in notebooks
- [ ] File operations work (CSV/MAT loading)
- [ ] Parallel processing functions available
- [ ] No permission errors with project files

## 🎯 Next Steps

1. **Start Learning**: Open `notebooks/00_environment_check.ipynb`
2. **Follow Sequence**: Work through notebooks 01-12 in order
3. **Practice**: Complete exercises in each notebook
4. **Apply Knowledge**: Tackle mini projects
5. **Build Portfolio**: Complete flagship project

## 📞 Getting Help

- **Check logs**: `docker-compose logs`
- **Documentation**: See `troubleshooting.md` for specific issues
- **Community**: Open GitHub issue with system details
- **Support**: Include error messages and system information

**Ready to master Octave!** 🎉
