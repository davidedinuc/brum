import argparse
import sys, os
from transformers import (
    AutoImageProcessor, Mask2FormerForUniversalSegmentation,
    SamModel, SamProcessor
    )
import torch
from pytorch3d.renderer import PerspectiveCameras
import numpy as np
import argparse, pickle, sys
from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path
from utils.dust3r_utils import ( 
     compute_sam,
     opencv_to_pytorch3d
     )
from utils.utils import set_seed

def main(args):

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'setting device to {device}')
   
    if args.segmentation:
        print('Loading sam model.... ')
        model_sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor_sam = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        print('...Done')

    for set in args.subset:
        print('processing {} data'.format(set))
        with open(Path(args.dust3r_path) / f'{set}_data.pickle' , 'rb') as file:
            data = pickle.load(file)

        images_name = [image for image in data.keys()]

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
                
                if args.segmentation:
                    torch.use_deterministic_algorithms(False)
                    input_points = [[[int(data[key]['rgb'].size[0] / 2), int(data[key]['rgb'].size[1] / 2)]]]

                    data[key]['segmentation'] = compute_sam(data[key]['rgb'], input_points, processor_sam, model_sam).detach().cpu().numpy()
                else:
                    data[key]['segmentation'] = np.ones_like(data[key]['depth'])
                
                pose = torch.from_numpy(data[key]['pose']).to(device)
                #invert the pose from c2w to w2c. PerspectiveCameras accepts w2c coordinates
                pose = torch.inverse(pose) 
                
                H, W = data[key]['depth'].shape[0], data[key]['depth'].shape[1]
                
                R, T = opencv_to_pytorch3d(pose)
                
                data[key]['torch_3d_pose'] = torch.cat([R, T.unsqueeze(1)], dim=1).cpu().numpy()
        
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
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=1518,
                        help='seed')

    parser.add_argument('--dust3r_path', type=str, default='/home/ddinucci/Desktop/datasets/tmp/dust3r_data',
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