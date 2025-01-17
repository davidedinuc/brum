"""
This script processes depth and image data for a given dataset. 
Functions:
    opencv_to_pytorch3d(opencv_pose):
        Converts an OpenCV pose matrix to a PyTorch3D pose matrix.
    main(args):
        Main function to process the dataset, align depth maps, and save the processed data.
Arguments:
    --seed: Seed for random number generation.
    --dust3r_path: Path to the dataset folder.
    --segmentation: Flag to indicate if segmentation should be processed.
    --subset: List of dataset subsets to process (train, test).
"""

import argparse
from utils.utils import set_seed
import pickle
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch 


def opencv_to_pytorch3d(opencv_pose):
        T = opencv_pose[:3, 3]
        R = opencv_pose[:3, :3]
        #from opencv to pytorch3d coordinate frame 
        R = R.permute(1, 0) 
        T[:2] *= -1
        R[:, :2] *= -1

        return R, T

def main(args):
    set_seed(args.seed)

    for set in args.subset:
        print('processing {} data'.format(set))
        with open(Path(args.dust3r_path) / f'{set}_data.pickle' , 'rb') as file:
            data = pickle.load(file)

        images_name = [image for image in data.keys()]

        imgs = []
        gt_depths = []
        for key in tqdm(images_name, desc='Processing duster data and load images'):

            data[key]['H'] = data[key]['depth'].shape[0]
            data[key]['W'] = data[key]['depth'].shape[1]
            
            image = Image.open(f'{args.dust3r_path}/images/{set}/' + key).resize((data[key]['W'],data[key]['H']))
            
            if np.array(image).shape[2] == 4:
                im_data = np.array(image.convert("RGBA"))

                bg = np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
            data[key]['rgb'] = image
            data[key]['segmentation'] = np.ones_like(data[key]['depth'])

            pose = torch.from_numpy(data[key]['pose'])
            additional_row = torch.tensor([[0, 0, 0, 1]], dtype=pose.dtype, device=pose.device)
            pose = torch.cat((pose, additional_row), dim=0)

            #invert the pose from c2w to w2c. PerspectiveCameras accepts w2c coordinates
            pose = torch.inverse(pose) 
            
            H, W = data[key]['depth'].shape[0], data[key]['depth'].shape[1]
            
            R, T = opencv_to_pytorch3d(pose)
            
            data[key]['torch_3d_pose'] = torch.cat([R, T.unsqueeze(1)], dim=1).cpu().numpy()
            
            if args.align_da and set == 'train':
                imgs.append(np.array(data[key]['rgb']))
                gt_depths.append(data[key]['depth'])

        intrinsic_focal_length = data[key]['focal']
        intrinsic_principal_point = (torch.tensor((W)/2), torch.tensor((H)/2))

        K = torch.zeros((3, 3))
        K[0, 0] = torch.tensor(intrinsic_focal_length[0])
        K[1, 1] = torch.tensor(intrinsic_focal_length[0])
        K[0, 2] = intrinsic_principal_point[0]
        K[1, 2] = intrinsic_principal_point[1]
        K[2, 2] = 1.0

        data['H'] = data[key]['depth'].shape[0]
        data['W'] = data[key]['depth'].shape[1]
        data['intrinsics'] = K.numpy()
        
        with open(f'{args.dust3r_path}/data_{set}_processed.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Data processed and saved in {args.dust3r_path}/data_{set}_processed.pickle')
        print('Done processing {} data'.format(set))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=1518,
                        help='seed')
    parser.add_argument('--dust3r_path', type=str, default='/home/prometeia.lan/dinuccid/datasets/carpatch/jeep/dust3r',
                        help='source folder')
    parser.add_argument('--segmentation', action='store_true')

    parser.add_argument(
        '--subset',
        type=str,
        nargs='+',
        choices=['train', 'test'],
        default=['train', 'test'],
    )

    args = parser.parse_args()

    main(args)