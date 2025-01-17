from utils.utils import set_seed
import argparse, json, cv2, os, shutil, pickle, sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

def read_poses(data):
    frames = data["frames"]
    poses = []
    for frame in (frames) :
        c2w = np.array(frame['transform_matrix'])[:3, :4]

        c2w[:, 1:3] *= -1  # [right up back] to [right down front]
        #pose_radius_scale = 1.5
        #c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale
        poses.append(c2w)

    return torch.from_numpy(np.array(poses))

def read_intrinsics(data):
    
    w = h = int(800)
    fx = fy = 0.5*800/np.tan(0.5*data['camera_angle_x'])

    K = np.float32([[fx, 0,  w/2],
                    [0, fy,  h/2],
                    [0,  0,   1]])

    return fx, (w/2, h/2)

def read_depths(source_folder, train_img_idxs):
    depths_path = source_folder + '/depth'
    depths = []
    for idx in train_img_idxs:
        depth = cv2.imread(f'{depths_path}/depth_{idx}.png', cv2.IMREAD_ANYDEPTH)/1000
        depths.append(depth)
    
    return np.array(depths)

def main(args):
    set_seed(args.seed)

    folder_train = Path(args.output_path) / 'images' / 'train'
    folder_train.mkdir(parents=True, exist_ok=True)
   
    folder_test = Path(args.output_path) / 'images' / 'test'
    folder_test.mkdir(parents=True, exist_ok=True)

    with open(f'{args.source_folder}/{args.train_folder}/{args.train_ids}', 'r') as file:
        train_img_idxs = [line.strip().split('_')[-1] for line in file]

    with open(f'{args.source_folder}/{args.train_folder}/transforms_train.json', 'r') as f:
        data_train = json.load(f)

    train_poses = read_poses(data_train)
    train_poses = [train_poses[int(idx)] for idx in train_img_idxs]

    depths = read_depths(f'{args.source_folder}/{args.train_folder}', train_img_idxs)

    source_images = [str(file_name) for file_name in os.listdir(f'{args.source_folder}/{args.train_folder}') if file_name.endswith((".jpg", ".png", ".JPG"))]
    source_images.sort()
    for i, img in enumerate(train_img_idxs):
        for source_image in source_images:
            if (img + '.png') in str(source_image).split('_')[-1]:
                print(f'{source_image} copied.')
                extension = source_image.split('.')[-1]
                shutil.copy(Path(args.source_folder) / args.train_folder / source_image, folder_train / f"{i:03d}.{extension}")

    with open(f'{args.source_folder}/test_key.txt', 'r') as file:
        test_img_idxs = [line.strip().split('_')[-1] for line in file]

    with open(f'{args.source_folder}/transforms_train.json', 'r') as f:
        data_test = json.load(f)

    depths_test = read_depths(f'{args.source_folder}/{args.train_folder}', test_img_idxs)

    test_poses = read_poses(data_test)
    test_poses = [test_poses[int(idx)] for idx in test_img_idxs]

    #copy selected train images to the respective folder
    print(f'Copying {len(test_img_idxs)} test images to {folder_test}...')
    for i, img in enumerate(test_img_idxs):
        for source_image in source_images:
            if (img + '.png') in str(source_image).split('_')[-1]:
                print(f'{source_image} copied.')
                extension = source_image.split('.')[-1]
                shutil.copy(Path(args.source_folder) / source_image, folder_test / f"{i:03d}.{extension}")

    images_train_paths = [str(folder_train / file_name) for file_name in os.listdir(folder_train) if file_name.endswith((".jpg", ".png", ".JPG"))]
    images_train_paths.sort()

    images_test_paths = [str(folder_test / file_name) for file_name in os.listdir(folder_test) if file_name.endswith((".jpg", ".png", ".JPG"))]
    images_test_paths.sort()

    images_paths = images_train_paths + images_test_paths
    images_paths.sort()

    train_data = {}
    test_data = {}

    fx, principal = read_intrinsics(data_train)

    for i, image in enumerate(images_train_paths):
        name = f'{image.split("/")[-1]}'
        train_data[name] = {}
        train_data[name]['focal'] = np.array([fx])
        train_data[name]['principal_points'] = np.array([principal])
        train_data[name]['pose'] = train_poses[i].detach().cpu().numpy()
        train_data[name]['confidence_mask'] = np.ones((800, 800))
        train_data[name]['depth'] = depths[i]
        #train_data[name]['3d_points'] = pts3d[i].detach().cpu().numpy()

    for i, image in enumerate(images_test_paths):
        name = f'{image.split("/")[-1]}'
        test_data[name] = {}
        test_data[name]['focal'] = np.array([fx])
        test_data[name]['principal_points'] = np.array([principal])
        test_data[name]['pose'] = test_poses[i].detach().cpu().numpy()
        test_data[name]['confidence_mask'] = np.ones((800, 800))
        test_data[name]['depth'] = depths_test[i] # np.ones((800, 800))

    with open(f'{args.output_path}/train_data.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{args.output_path}/test_data.pickle', 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=1518,
                        help='seed')
    parser.add_argument('--output_path', type=str, default='/home/prometeia.lan/dinuccid/datasets/carpatch/jeep/dust3r',
                        help='destination folder')
    parser.add_argument('--source_folder', type=str, default='/home/prometeia.lan/dinuccid/datasets/carpatch/jeep/images',
                        help='source folder')

    parser.add_argument('--train_folder', type=str, default='')
    parser.add_argument('--train_ids', type=str, default='train_key_8.txt',)
    
    args = parser.parse_args()
    main(args)