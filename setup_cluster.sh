#!/bin/bash
# Setup script for nuel-sim on HPC cluster

# Install miniconda if not already installed
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
fi

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Remove existing environment if it exists
conda env remove --name myenv -y 2>/dev/null || true

# Create conda environment
conda create --name myenv python=3.11 -y
conda activate myenv

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other requirements
pip install ray[rllib] pygame matplotlib numpy scipy

echo "Environment setup complete. Use 'sbatch run_cluster.sh' to submit jobs."