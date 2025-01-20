# BRUM: Robust 3D Vechicle Reconstruction from 360° Sparse Images 


<p align="center">
  <img src="./imgs/brum_logo.png" alt="Logo" width="400">
</p>

installation 

# Install PyTorch3D
```bash
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

# Install gs dependencies
```bash 
pip install -r ../requirements.txt
pip install -e ../gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e ../gaussian-splatting/submodules/simple-knn
```

# Install dust3r dependencies
```bash
pip install -r ../dust3r/requirements.txt
```
