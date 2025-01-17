#!/bin/bash
cd ..
device=1
bus=2
n_images=12
int_factor=0.00008
th_dust3r=0.5

source_path=/home/prometeia.lan/dinuccid/datasets/seta/bus${bus}/images
dust3r_path=/home/prometeia.lan/dinuccid/datasets/seta/bus${bus}/dust3r/dust3r_${n_images}_images
dataset_path=/home/prometeia.lan/dinuccid/datasets/seta/bus${bus}/${n_images}_images
CUDA_VISIBLE_DEVICES=$device python3 run_dust3r.py --output_path $dust3r_path --source_folder $source_path --train_ids train_key_12.txt
CUDA_VISIBLE_DEVICES=$device python3 process_dust3r_data.py --dust3r_path $dust3r_path --segmentation

dataset_path=/home/prometeia.lan/dinuccid/datasets/seta/bus${bus}/${n_images}/${n_images}_images_int_${int_factor}_th_${th_dust3r}_10000_pytorch_weights
CUDA_VISIBLE_DEVICES=$device python3 dataset_creation.py --dust3r_path $dust3r_path  --folder_path $dataset_path --int_factor $int_factor --th_dust3r $th_dust3r --pytorch_weights #--distance_weights 

cd gaussian-splatting

#output_path=./output/bus${bus}/${n_images}_images/${n_images}_int_${int_factor}_th_${th_dust3r}_10000_pytorch_weights
output_path=./output/bus${bus}/${n_images}_images/${n_images}_gt
CUDA_VISIBLE_DEVICES=$device python train.py -s ${dataset_path} -m ${output_path} --torch_data -r 1 --mask_loss --iterations 10000 --test_iterations 5000 --port 7070 
CUDA_VISIBLE_DEVICES=$device python render.py -m ${output_path} --skip_train --torch_data --eval
CUDA_VISIBLE_DEVICES=$device python metrics.py -m ${output_path}
cd ..
