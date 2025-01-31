# BRUM: Robust 3D Vechicle Reconstruction from 360° Sparse Images 


<p align="center">
  <img src="./imgs/brum_logo.png" alt="Logo" width="400">
</p>

## Installation

1. Clone repository and setup environment.
The repository contains submodules, thus please check it out with:
```bash
git clone https://github.com/davidedinuc/brum.git --recursive
conda create -n brum python=3.9
conda activate brum
```

1. Install PyTorch3D.
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
```

2. Install gs dependencies
```bash
pip install gaussians-splatting/submodules/diff-gaussian-rasterization
pip install gaussians-splatting/submodules/simple-knn
```

3. Install dust3r dependencies.
```bash
pip install -r ./dust3r/requirements.txt
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

