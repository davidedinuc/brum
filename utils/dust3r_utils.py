from pathlib import Path
import os, shutil
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import cv2

def prepare_dust3r_data(args):

    source_folder = Path(args.source_folder) #contains the folder with all the dataset images
    source_images = [str(file_name) for file_name in os.listdir(source_folder) if file_name.endswith((".jpg", ".png", ".JPG"))]
    source_images.sort()
    
    #create folder for the train and test images
    folder_train = Path(args.output_path) / 'images' / 'train'
    folder_train.mkdir(parents=True, exist_ok=True)

    folder_test = Path(args.output_path) / 'images' / 'test'
    folder_test.mkdir(parents=True, exist_ok=True)
    
    with open(f'{args.source_folder}/{args.train_ids}', 'r') as file:
         train_img_idxs = [line.strip() for line in file]
    
    with open(f'{args.source_folder}/test_key.txt', 'r') as file:
         test_img_idxs = [line.strip() for line in file]

    #copy selected train images to the respective folder
    print(f'Copying {len(train_img_idxs)} train images to {folder_train}...')

    for i, img in enumerate(train_img_idxs):
        for source_image in source_images:
            if img in str(source_image):
                print(f'{source_image} copied.')
                extension = source_image.split('.')[-1]
                shutil.copy(str(source_folder / source_image), folder_train / f"{i:03d}.{extension}")

    print('...Done!')
    images_train_paths = [str(folder_train / file_name) for file_name in os.listdir(folder_train) if file_name.endswith((".jpg", ".png", ".JPG"))]
    images_train_paths.sort()

    #copy selected train images to the respective folder
    print(f'Copying {len(test_img_idxs)} test images to {folder_test}...')
    for source_image in source_images:
        if any(img in str(source_image) for img in test_img_idxs):
            print(f'{source_image} copied.')
            shutil.copy(str(source_folder / source_image), folder_test / source_image)

    print('...Done!')
    images_test_paths = [str(folder_test / file_name) for file_name in os.listdir(folder_test) if file_name.endswith((".jpg", ".png", ".JPG"))]
    images_test_paths.sort()

    return images_train_paths, images_test_paths

def compute_sam(pil_img, input_points, processor, model):

    inputs = processor(pil_img, return_tensors="pt").to(model.device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    inputs = processor(pil_img, input_points=input_points, return_tensors="pt").to(model.device)
    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    
    mask_areas = [mask.sum().item() for mask in masks[0][0]]
    # Get the index of the mask with the largest area
    largest_mask_index = mask_areas.index(max(mask_areas))
    # Get the largest mask
    largest_mask = masks[0][0][largest_mask_index]

    return largest_mask

def opencv_to_pytorch3d(opencv_pose):
        T = opencv_pose[:3, 3]
        R = opencv_pose[:3, :3]
        #from opencv to pytorch3d coordinate frame 
        R = R.permute(1, 0) 
        T[:2] *= -1
        R[:, :2] *= -1

        return R, T