# BRUM: Robust 3D Vechicle Reconstruction from 360° Sparse Images 


<p align="center">
  <img src="./imgs/brum_logo.png" alt="Logo" width="400">
</p>

## Installation

1. Install PyTorch3D.
```bash
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

2. Install gs dependencies
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt
```

3. Install dust3r dependencies.
```bash
pip install -r ../dust3r/requirements.txt

```

## Download data
To download out BRUM-dataset, please follow the links below:

[Real data](https://ailb-web.ing.unimore.it/publicfiles/drive/brum-dataset/real_bus.zip)    
[Synthetic data](https://ailb-web.ing.unimore.it/publicfiles/drive/brum-dataset/synt_bus.zip)    
[Blend Files](https://ailb-web.ing.unimore.it/publicfiles/drive/brum-dataset/blend_files.zip)    

## Usage

```bash
#For real data setup ./scripts/run_pipeline_real.sh 
#For synthetic data setup ./scripts/run_pipeline_synthetic.sh 
cd scripts
./run_pipeline_real.sh
./run_pipeline_synthetic.sh
```

