"""
This script runs the DUSt3R model for 3D stereo image processing and global alignment.
Functions:
    main_dust3r(args): Main function to run the DUSt3R model with specified arguments.
Arguments:
    --seed (int): Seed for random number generation. Default is 1518.
    --output_path (str): Destination folder for output files.
    --source_folder (str): Source folder containing input images.
    --train_ids (str): File containing training image IDs. Default is 'train_key_8.txt'.
    --size (int): Size to which input images will be resized. Default is 512.
Usage:
    Run the script with the desired arguments to process images using the DUSt3R model and save the results.
Example:
    python run_dust3r.py --seed 1234 --output_path /path/to/output --source_folder /path/to/images --train_ids train_ids.txt --size 256
"""
import sys
sys.path.append('./dust3r')
import numpy as np
import pickle
import os, shutil
from pathlib import Path
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import argparse
from utils.dust3r_utils import ( 
            prepare_dust3r_data  )
import torch
from utils.utils import set_seed



def main_dust3r(args):

    set_seed(args.seed)

    images_train_paths, images_test_paths = prepare_dust3r_data(args)
    images_paths = images_train_paths + images_test_paths
    images_paths.sort()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    print(f'Loading model {model_name}...')
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    print('...Done')
    size = args.size
    images = load_images(images_paths , size=size)
    pairs = make_pairs(images, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)

    #FOCALE FISSATA PER OGNI IMMAGINE
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    H = images[0]['img'].squeeze(0).shape[1]
    W = images[0]['img'].squeeze(0).shape[2]

    fx, fy = max(W, H), max(W, H)

    focals = torch.tensor([fx,fx,fx,fx,fx,fx,fx,fx]).float()

    scene.preset_focal(focals)
    scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    focals = scene.get_focals()
    pts3d = scene.get_pts3d()

    data = {}
    pts3d = scene.get_pts3d()
    scene.min_conf_thr = 0


    for i, image in enumerate(images_paths):
        name = f'{image.split("/")[-2]}/{image.split("/")[-1]}'
        data[name] = {}
        data[name]['focal'] = scene.get_focals()[i].detach().cpu().numpy()
        data[name]['principal_points'] = scene.get_principal_points()[0].cpu().numpy()
        data[name]['pose'] = scene.get_im_poses()[i].detach().cpu().numpy()
        data[name]['confidence_mask'] = scene.im_conf[i].cpu().numpy()
        data[name]['depth'] = scene.get_depthmaps()[i].detach().cpu().numpy()
        data[name]['3d_points'] = pts3d[i].detach().cpu().numpy()

    train_data = {}
    test_data = {}
    for key in list(data.keys()):
        if 'train' in key:
            train_data[key.split('/')[-1]] = data[key]
        else:
            test_data[key.split('/')[-1]] = data[key]

    with open(f'{args.output_path}/train_data.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{args.output_path}/test_data.pickle', 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Dust3r Done!')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=1518,
                        help='seed')
    parser.add_argument('--output_path', type=str, default='/home/prometeia.lan/dinuccid/datasets/tmp',
                        help='destination folder')
    parser.add_argument('--source_folder', type=str, default='/home/prometeia.lan/dinuccid/datasets/ford_focus/images',
                        help='source folder')
    parser.add_argument('--train_ids', type=str, default='train_key_8.txt',)
    parser.add_argument('--size', type=int, default=512,)
    
    args = parser.parse_args()

    main_dust3r(args)

