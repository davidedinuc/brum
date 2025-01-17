
"""
This script processes 3D point cloud data and generates interpolated images and masks for training and testing.
Functions:
    main(args): Main function to process the data and generate images, masks, and point clouds.
Arguments:
    --seed (int): Seed for random number generation. Default is 1518.
    --dust3r_path (str): Path to the source folder containing the data.
    --folder_path (str): Path to the folder where the output will be saved.
    --th_dust3r (float): Confidence threshold for Dust3D. Default is 0.
    --ppp (int): Points per pixel. Default is 16.
    --int_factor (float): Interpolation factor. Default is 0.2.
    --pytorch_weights (bool): Flag to use PyTorch weights. Default is False.
    --save_depths (bool): Flag to save depth information. Default is False.
    --radius (float): Radius for rendering. Default is 0.01.
"""

from PIL import Image
import numpy as np
import os, pickle
from pathlib import Path
from pytorch3d.structures import Pointclouds
import torch
import argparse
from pytorch3d.renderer import (
    PerspectiveCameras,
)
from tqdm import tqdm
from utils.utils import set_seed
from utils.pytorch3d_utils import (
                        xyz_multiple_cameras, 
                        save_pointcloud_with_normals,
                        merge_pointclouds,
                        render_image_new_cam,
                        render_image_with_slerp
                    )

def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    folder_path = Path(args.folder_path)
    os.makedirs(folder_path, exist_ok=True)

    dust3r_path = args.dust3r_path
    data_set = 'train' #test or train
    duster_data_path = f'{dust3r_path}/data_{data_set}_processed.pickle'

    with open(duster_data_path, 'rb') as file:
        data_train = pickle.load(file)
        
    images_name = [image for image in data_train.keys() if (image != 'H' and image != 'W' and image != 'intrinsics')]
    images_name.sort()

    rotations = torch.from_numpy(np.stack([data_train[name]['torch_3d_pose'][:3,:3] for name in images_name])).to(device)
    translations = torch.from_numpy(np.stack([data_train[name]['torch_3d_pose'][:3,3] for name in images_name])).to(device)
    focals = torch.stack([torch.from_numpy(data_train[name]['focal']) for name in images_name]).to(device)
    H, W = data_train['H'], data_train['W']

    cameras = PerspectiveCameras(R=rotations.squeeze(0), T=translations.squeeze(0), 
                                focal_length=focals.float(), principal_point=(((W)/2, (H)/2),), 
                                image_size=((H, W),), device=device, in_ndc=False)
    
    align_depth_anything = args.align_depth_anything
    apply_mask = True

    threshold = args.th_dust3r #Dust3d confidence threshold 
    confidence = torch.stack([torch.from_numpy(data_train[image_name]['confidence_mask']).float() for image_name in images_name]).to(device)
    segmentation = torch.stack([torch.from_numpy(data_train[image_name]['segmentation']).float() for image_name in images_name]).to(device)

    confidence_mask = (confidence > threshold)
    mask = confidence_mask * segmentation.bool()

    if not apply_mask:
        mask = torch.ones_like(mask)

    duster_depths = torch.stack([torch.from_numpy(data_train[image_name]['depth']).float() for image_name in images_name]).to(device)

    xy_depth_world = xyz_multiple_cameras(duster_depths, cameras) * mask.view(duster_depths.shape[0],-1).unsqueeze(-1) # * masks.unsqueeze(-1) 
    img_tensor = torch.stack([torch.from_numpy(np.array(data_train[name]['rgb'])).reshape(-1, 3).float().to(device) / 255 for name in images_name], dim=0) #* mask.view(4,-1).unsqueeze(-1)

    pcs = []
    for i in range(len(images_name)):
        pcs.append(Pointclouds(points=xy_depth_world[i].view(-1,3).unsqueeze(0), features=img_tensor[i].view(-1,3).unsqueeze(0)))
    global_pc = merge_pointclouds(pcs)

    save_pointcloud_with_normals(img_tensor, xy_depth_world, mask, Path(folder_path) / 'sparse_colmap/0')

    combinations = [(i, (i + 1) % len(images_name)) for i in range(len(images_name))]
    intermediate_images = 40

    count=0
    radius=args.radius 
    camera_poses_train = {'poses' :{}}
    int_factor = args.int_factor
    ppp = args.ppp

    render_folder = folder_path / 'images/train/images'
    masks_folder = folder_path / 'images/train/masks'
    if align_depth_anything:
        depths_folder = folder_path / 'images/train/depths'
        os.makedirs(depths_folder, exist_ok=True)
    weights_folder = folder_path / 'images/train/weights'
    local_masks_folder = folder_path / 'images/train/local_mask'

    os.makedirs(weights_folder, exist_ok=True)
    os.makedirs(render_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    total_steps = len(combinations) * intermediate_images
    pbar = tqdm(total=total_steps)
    for i, j in combinations:
        for interpolation_factor in np.linspace(0.0, 1.0, intermediate_images):
            if interpolation_factor == 0:

                image_array = np.array(data_train[images_name[i]]['rgb']) * data_train[images_name[i]]['segmentation'][..., None]
                Image.fromarray((image_array).astype(np.uint8)).save(f'{render_folder}/{count:03}.png')

                gt_mask = np.ones((image_array.shape[0],image_array.shape[1])) 

                np.save(f'{masks_folder}/{count:03}.npy', gt_mask)
                weights = np.ones_like(gt_mask)
                np.save(f'{weights_folder}/{count:03}.npy', weights)
                Image.fromarray((gt_mask * 255).astype(np.uint8)).save(f'{local_masks_folder}/{count:03}.png')

                rt = torch.cat([cameras[i].R[0], cameras[i].T[0].unsqueeze(1)], dim=1)
                camera_poses_train['poses'][f'{count:03}.png'] = rt.cpu().numpy()
                count += 1

            elif interpolation_factor == 1:
                break
            else:

                if interpolation_factor <= int_factor:
                    pc = pcs[i]
                elif interpolation_factor >= 1 - int_factor:
                    pc = pcs[j]

                else:
                    continue

                render_img, render_mask, _, int_camera, render_weights = render_image_with_slerp(cameras, interpolation_factor, radius, [i, j], ppp=ppp, pc=pc)
                if align_depth_anything:
                    _, render_mask_da, render_dept_da  = render_image_new_cam(int_camera, da_pc, radius=radius, weights_precision='coarse')
                    depth_da = render_dept_da[0].detach().cpu().numpy() * render_mask_da[0].detach().cpu().numpy()
                    np.save(f'{depths_folder}/{count:03}.npy', depth_da)
  
                _, global_mask, _, _, _ = render_image_with_slerp(cameras, interpolation_factor, radius, [i, j], ppp=ppp, pc=global_pc)

                xor_mask = 1 - (global_mask[0].detach().cpu().numpy().astype(np.uint8) ^ render_mask[0].detach().cpu().numpy().astype(np.uint8))

                if args.pytorch_weights:
                    weights = (render_weights[0].detach().cpu().numpy() + (1 - global_mask[0].detach().cpu().numpy()))
                else:
                    weights = np.ones_like(xor_mask)

                Image.fromarray((render_img[0].detach().cpu().numpy() * 255).astype(np.uint8)).save(f'{render_folder}/{count:03}.png')
                np.save(f'{masks_folder}/{count:03}.npy', xor_mask)
                np.save(f'{weights_folder}/{count:03}.npy', weights)

                rt = torch.cat([int_camera.R[0], int_camera.T[0].unsqueeze(1)], dim=1)
                camera_poses_train['poses'][f'{count:03}.png'] = rt.cpu().numpy()
                count += 1

            pbar.update(count)

    pbar.close()

    camera_poses_train['intrinsics'] = data_train['intrinsics']

    data_set = 'test' 
    camera_poses_test = {}

    render_folder = folder_path / f'images/{data_set}/images'
    os.makedirs(render_folder, exist_ok=True)

    duster_data_path = f'{dust3r_path}/data_{data_set}_processed.pickle'
    with open(duster_data_path, 'rb') as file:
        data_test = pickle.load(file)
        
    images_name_test = [image for image in data_test.keys() if (image != 'H' and image != 'W' and image != 'intrinsics')]
    images_name_test.sort()
    for image_name in images_name_test:
        image_array = np.array(data_test[image_name]['rgb']) * data_test[image_name]['segmentation'][..., None] 
        image_name = image_name.split('.')[-2]
        Image.fromarray((image_array).astype(np.uint8)).save(f'{render_folder}/{image_name}.png')

    camera_poses_test['poses'] = {image.replace('jpg','png').replace('JPG', 'png'): data_test[image]['torch_3d_pose'] for image in images_name_test}

    camera_poses_test['intrinsics'] = data_test['intrinsics']

    if args.save_depths:
        with open(f'{dust3r_path}/data_test_processed.pickle', 'rb') as file:
            data_test = pickle.load(file)

        camera_poses_test['depths'] = {image.replace('jpg','png').replace('JPG', 'png'): data_test[image]['depth'] for image in images_name_test}
        camera_poses_train['depths'] = {image.replace('jpg','png').replace('JPG', 'png'): 
                                       data_train[image]['depth'] for image in images_name}

    with open(f'{folder_path}/images/test/camera_test.pickle', 'wb') as f:
        pickle.dump(camera_poses_test, f)

    with open(f'{folder_path}/images/train/camera_train.pickle', 'wb') as f:
        pickle.dump(camera_poses_train, f)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=1518,
                        help='seed')
    parser.add_argument('--dust3r_path', type=str, default='',
                        help='source folder')
    parser.add_argument('--folder_path',  type=str, default='')
    parser.add_argument('--th_dust3r', type=float, default=0.)
    parser.add_argument('--ppp', type=int, default=16)

    parser.add_argument('--int_factor', type=float, default=0.2)

    parser.add_argument('--pytorch_weights', action='store_true', default=False)
    parser.add_argument('--save_depths', action='store_true', default=False)

    parser.add_argument('--radius', type=float, default=0.01)

    args = parser.parse_args()

    main(args)