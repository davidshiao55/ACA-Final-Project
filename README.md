# Advanced Computer Architecture Final Project Report
>Our Project is to reproduce the [Smart-Infinity paper](https://arxiv.org/abs/2403.06664). [[Repo](https://github.com/AIS-SNU/Smart-Infinity)]
## Experiment
### Envoriment
#### Hardware
1. NVIDIA RTX 3090-GPU (24GB) 
2. 16-core Intel I7 12700 CPU 
3. 128GB RAM
4. Samsung SmartSSD
#### Software
1. Ubuntu 20.04 with Linux kernel 5.4.0-164
2. CUDA 11.6
3. PyTorch 1.13
4. DeepSpeed v0.9.3

![Alt text](images/Heterogeneous%20System.png)
### Challenges
#### System Support
1. The SmartSSD product line has been discontinued, making it difficult to find systems with compatible configurations or support.
$\rightarrow$ We find the compatible system configuration by trial and error.
#### Cooling
2. The SmartSSD is designed for installation in servers with controlled airflow for effective cooling, which makes local installation in a standard PC challenging due to potential overheating.
$\rightarrow$ We use a external fan to cooldown the SmartSSD.
### Result
![Alt text](images/Elapsed%20Time%20Breakdown%20per%20Iteration.png)
We profile the time breakdown of 3 configuration
1. CPU only : optimizer states is stored in the cpu memory and updated by the cpu. 
2. NVMe : optimizer states is stored in NVMe SSD, and fetched to cpu to update.
3. SmartSSD : optimizer states is sotred in NVMe SSD, and fetched to fpga to update.
## Analysis
As we only have one SmartSSD device we conduct a scalability analysis to verify the result from the paper.
From configuration 1 and 2, we can derived that most of the time spent during optimizer step is used on IO of fetching and offloading optimizer state.
As there are no data-dependencies between each parameter in optimizer state, the operation is highly scalable.
### Time Breakdown
We further classify the time during optimizer step into parallelizable and weakly-parallelizable. 
![Alt text](images/Data%20Movement.png)
![Alt text](images/Data%20movement%20Time.png)
from the time breakdoown above:
- parallelizable : scale linearly with # of CSDs
    1. FPGA Computation
    2. NVMe->FPGA Communication
    3. FPGA->NVMe Communication
- weakly-parallelizable : scaling upper bound by PCIe bandwidth
    1. FPGA->CPU Communication

### Scaling Upperbound
From the above analysis we can derive the scaling upperbound to be:
![Alt text](images/Scaling%20Upperbound.png)
and plotting it against the # of CSD devices.
![Alt text](images/Optimizer%20Step%20Speedup.png)
![Alt text](images/Speedup%20of%20Total%20Step%20Time.png)
We derive same scalability from the paper which test with real machine.


## Installation and Setup
### SmartSSD Installation
#### Operating System
Ubutu 20.04 with Linux kernel 5.4.0-164
#### Kernel Installation
```bash=
# install kernel
sudo apt update
sudo apt install linux-image-5.4.0-164-generic linux-headers-5.4.0-164-generic
sudo update-grub
```
#### FPGA Driver Installation
```bash=
# install FPGA driver from source
sudo apt install linux-source
cd /usr/src
tar -xf linux-source-*.tar.bz2

sudo cp -r /usr/src/linux-source-5.4.0/drivers/fpga /usr/src/linux-headers-5.4.0-164-generic/drivers/
cd /usr/src/linux-headers-5.4.0-164-generic
make menuconfig
# Device Drivers  --->
#  [*] FPGA Configuration Framework
make M=drivers/fpga modules

sudo cp drivers/fpga/*.ko /lib/modules/$(uname -r)/kernel/drivers/fpga/
sudo depmod
sudo modprobe fpga_mgr
```

#### Install XRT
```bash=
sudo apt install ./xrt*.deb
```
#### Install Development Target Platform
```bash=
# Install Development Target Platform
sudo apt install ./xilinx-u2-gen3x4-xdma-gc-2-202110-1-dev*.deb 
sudo apt install ./xilinx-u2-gen3x4-xdma-gc-validate_2*.deb 
sudo apt install ./xilinx-u2-gen3x4-xdma-gc-base_2*.deb
```
#### Flash smartSSD
```bash=
# Flash the SmartSSD with current IDE platform#
sudo /opt/xilinx/xrt/bin/xbmgmt program --base --image /opt/xilinx/firmware/u2/gen3x4-xdma-gc/base/partition.xsabin /opt/xilinx/firmware/u2/gen3x4-xdma-gc/base/partition.xsabin --flash-type spi --device <xclmgmt BDF>
```
#### Mount SSD
```bash=
sudo mkdir /mnt/smartssd1
sudo gdisk /dev/nvme2n1
# n -> w
sudo mkfs.ext4 /dev/nvme2n1p1

sudo mount /dev/nvme2n1p1 /smartssd1
sudo blkid /dev/nvme2n1p1
# Note down the UUID value (e.g., UUID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").
sudo vim /etc/fstab
# UUID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" /smartssd1 ext4 defaults 0 2

sudo chown $USER:$USER /mnt/smartssd1
```

### Smart-Infinity
#### Cuda Installation
```bash=
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-6
```
#### Conda Installation
```bash=
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n aca python=3.9 -y
conda activate aca
pip install torch==1.13.1+cu116 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

#### Additional Installation
```bash=
conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge oneccl-devel
```

#### Deepspeed installation
```bash=
export ENV=aca
export CUDA_HOME=/usr/local/cuda-11.6/

git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
git checkout v0.9.3

DS_BUILD_OPS=1 \
DS_BUILD_SPARSE_ATTN=0 \
DS_BUILD_UTILS=0 \
DS_BUILD_CPU_ADAM=0 \
DS_BUILD_FUSED_ADAM=0 \
DS_BUILD_FUSED_LAMB=0 \
DS_BUILD_RANDOM_LTD=0 \
DS_BUILD_TRANSFORMER=0 \
DS_BUILD_TRANSFORMER_INFERENCE=0 \
DS_BUILD_QUANTIZER=0 \
DS_BUILD_SPATIAL_INFERENCE=0 \
DS_BUILD_AIO=0 \
pip install . --no-cache-dir

conda install -c conda-forge regex -y

git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Backup for original deepseed
cp -r ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/deepspeed  ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/_deepspeed

# SmartInfinity
rm -rf ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/deepspeed

ln -s ~/Smart-Infinity/deepspeed ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/deepspeed
```
#### Install Vitis
```bash=
sudo apt-get install libtinfo5
sudo apt install libncurses5
# https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/archive-vitis.html
sudo ./Xilinx_Unified_2023.1_0507_1903_Lin64.bin
```

#### Generate binary file
```bash=
sudo apt install opencl-clhpp-headers
make xclbin LAB=run1 #Adam only
cp krnl_vadd.hw.xilinx_u2_gen3x4_xdma_gc_2_202110_1.xclbin adam.xclbin
make host LAB=run1 # Adam only
./host
cp adam.xclbin ~/bins/adam.xclbin
```

## Run Smart-Infinity
```bash=
./run_smartinfinity.sh
```
See `DeepSpeedExample/example/ds_zero_stage_infinity-nvme.json` for important hyperparameters for using SmartInfinity.

## Reference
[SmartSSD Installation Guide](https://syh.one/posts/bu-ms-thesis-accelerating-gnn-training-with-smartssd/installation/)
[SmartSSD Official Doc](https://docs.amd.com/v/u/en-US/ug1382-smartssd-csd)
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
[cuda-toolkit](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)
[Smart-Inifinity](https://github.com/AIS-SNU/Smart-Infinity)
https://github.com/NVIDIA/apex/issues/550
https://github.com/NVIDIA/apex/issues/990
https://github.com/deepspeedai/DeepSpeed/issues/5653
https://github.com/deepspeedai/DeepSpeed/blob/6ea44d02c674393c524ada811ea376c55438a913/requirements/requirements-dev.txt#L4
https://adaptivesupport.amd.com/s/question/0D54U00008jzriDSAQ/vivado-20241-instalation-gets-stuck-in-generating-installed-device-list?language=en_US
