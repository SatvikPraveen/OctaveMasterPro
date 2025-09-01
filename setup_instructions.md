# File location: OctaveMasterPro/setup_instructions.md

# 🛠️ Setup Instructions for OctaveMasterPro

## 🐳 Docker Setup (Recommended)

### Prerequisites

- Docker Desktop installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (included with Docker Desktop)
- 4GB+ RAM available
- 10GB+ disk space

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/SatvikPraveen/OctaveMasterPro.git
cd OctaveMasterPro

# 2. Initialize project
chmod +x init_project.sh
./init_project.sh

# 3. Start services
docker-compose up
```

### Access Points

- **Jupyter Lab**: http://localhost:8888
- **Alternative Port**: http://localhost:8889
- **CLI Access**: `docker-compose --profile cli run octave-cli`

## 🖥️ Local Installation

### Ubuntu/Debian

```bash
# Install Octave and dependencies
sudo apt update
sudo apt install octave octave-parallel octave-statistics octave-image octave-signal

# Install Python and Jupyter
sudo apt install python3 python3-pip jupyter-notebook

# Install Octave kernel for Jupyter
pip3 install octave_kernel
python3 -m octave_kernel install --user

# Launch Jupyter
jupyter lab
```

### macOS (Homebrew)

```bash
# Install Octave
brew install octave

# Install Python and Jupyter
brew install python3
pip3 install jupyter jupyterlab octave_kernel

# Install Octave kernel
python3 -m octave_kernel install --user

# Launch Jupyter
jupyter lab
```

### Windows

1. **Install Octave**: Download from [octave.org](https://octave.org/download)
2. **Install Python**: Download from [python.org](https://python.org/downloads)
3. **Install Jupyter**:
   ```cmd
   pip install jupyter jupyterlab octave_kernel
   python -m octave_kernel install --user
   ```
4. **Launch**: `jupyter lab`

## 🔧 Verification Steps

### 1. Test Octave Installation

```bash
octave --version
```

Expected: GNU Octave version 6.0+

### 2. Test Jupyter Kernel

```bash
jupyter kernelspec list
```

Expected: `octave` kernel listed

### 3. Test Packages

Open Octave and run:

```octave
pkg list
```

Required packages: statistics, image, signal, parallel, optim

### 4. Test Notebook

1. Open `notebooks/00_environment_check.ipynb`
2. Run all cells
3. Verify plots display correctly

## 🚨 Troubleshooting

### Docker Issues

**Port already in use**:

```bash
# Check what's using port 8888
lsof -i :8888

# Use alternative port
docker-compose up --scale octave-master=0
docker run -p 8889:8888 octave-master
```

**Permission denied**:

```bash
sudo chmod +x init_project.sh
sudo chown -R $USER:$USER .
```

**Container won't start**:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Local Installation Issues

**Octave kernel not found**:

```bash
# Reinstall kernel
pip3 uninstall octave_kernel
pip3 install octave_kernel
python3 -m octave_kernel install --user --force
```

**Package missing**:

```bash
# In Octave command line
pkg install -forge statistics
pkg install -forge image
pkg install -forge signal
pkg install -forge parallel
```

**Plotting issues**:

```bash
# Install graphics backend
sudo apt install libqt5gui5  # Ubuntu
brew install qt5            # macOS
```

### Performance Issues

**Slow startup**:

- Close unnecessary applications
- Ensure 4GB+ RAM available
- Use SSD storage if possible

**Jupyter slow**:

```bash
# Clear cache
jupyter --paths
rm -rf ~/.jupyter
```

**Memory issues**:

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory → 4GB+
```

## 📊 System Requirements

### Minimum

- **CPU**: 2 cores
- **RAM**: 2GB
- **Storage**: 5GB
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04+

### Recommended

- **CPU**: 4+ cores
- **RAM**: 4GB+
- **Storage**: 10GB+ (SSD)
- **OS**: Latest stable versions

## 🌐 Network Configuration

### Firewall Settings

Allow ports: 8888, 8889

### Proxy Settings

```bash
# Docker with proxy
docker-compose up --build \
  --build-arg HTTP_PROXY=$http_proxy \
  --build-arg HTTPS_PROXY=$https_proxy
```

## 🔄 Updates

### Update Docker Images

```bash
docker-compose down
docker-compose pull
docker-compose up --build
```

### Update Local Installation

```bash
# Update Octave packages
octave --eval "pkg update"

# Update Python packages
pip3 install --upgrade jupyter jupyterlab octave_kernel
```

## 🆘 Getting Help

1. **Check logs**: `docker-compose logs octave-master`
2. **Reset environment**: `docker-compose down -v && docker-compose up --build`
3. **Open issue**: Include system info, error messages, and logs
4. **Community**: Join discussions in GitHub issues

## ✅ Post-Setup Checklist

- [ ] Docker containers start successfully
- [ ] Jupyter Lab accessible at localhost:8888
- [ ] Octave kernel available in Jupyter
- [ ] Environment check notebook runs without errors
- [ ] Plots display correctly
- [ ] All required packages installed
- [ ] File permissions correct
- [ ] Network access working

**Ready to start learning!** 🚀 Open `notebooks/00_environment_check.ipynb` to begin.
