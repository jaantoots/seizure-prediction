#!/bin/bash

install_cuda() {
    # Check for cuDNN file
    if [ ! -f cudnn-8.0-linux-x64-v5.1.tgz ]; then
        echo "Download cudnn-8.0-linux-x64-v5.1.tgz." >&2
        exit 1
    fi

    # Update everything first and install some stuff
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get install -y linux-image-extra-virtual linux-source
    sudo apt-get install -y linux-headers-3.13.0-96-generic

    # CUDA
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo apt-get update
    sudo apt-get install -y cuda

    # This is necessary to avoid setting LD_* environment variables
    sudo sh -c "sudo echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda.conf"
    sudo ldconfig /usr/local/cuda/lib64

    # Install cuDNN
    tar -xvf cudnn-8.0-linux-x64-v5.1.tgz
    sudo cp -P cuda/lib64/* /usr/local/cuda/lib64/
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/

    # Just checking
    sudo apt-get update
    sudo apt-get upgrade -y

    # Clean up
    if [ ! -d install ]; then
        mkdir install
    fi
    rm -rf cuda/
    mv cud* install/

    echo "Continue after reboot..." >&2
    sudo reboot
}

hash nvidia-smi >/dev/null 2>&1 || install_cuda

# Check CUDA
nvidia-smi

# Install Torch
sudo apt-get install git
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch || exit 1
bash install-deps
./install.sh
. ~/.bashrc

# Install rocks
sudo apt-get install libmatio2
luarocks install matio
luarocks install argparse
luarocks install json
luarocks install rnn
