#!/bin/bash
set -e

echo "================================================="
echo "BISK RFv4 - NVIDIA / CUDA Runtime Setup"
echo "================================================="

sudo apt update
# Install the tool and ckeck the recommended and install it manually:
#sudo apt install ubuntu-drivers-common
#ubuntu-drivers devices
# or auto install the recommended one:
sudo ubuntu-drivers install
# Core build tools
sudo apt install -y \
    build-essential \
    python3-dev \
    python3-venv \
    git \
    curl \
    wget

# NVIDIA CUDA runtime libraries required by ONNX Runtime GPU
sudo apt install -y \
    libcublas12 \
    libcublaslt12 \
    libcurand10 \
    libcufft11 \
    libcufftw11 \
    libcusparse12

# cuDNN
sudo apt install -y nvidia-cudnn

echo
echo "Installed CUDA Runtime Libraries:"
ldconfig -p | grep -E "cublas|cudnn|curand|cufft|cusparse" || true

echo
echo "GPU:"
nvidia-smi || true

echo
echo "Setup completed."