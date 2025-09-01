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
