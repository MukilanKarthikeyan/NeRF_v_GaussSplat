import os
from typing import Dict, List, Tuple
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import csv
import numpy as np

import json


class SpatialImg(data.Dataset):

    train_folder = "train"
    test_folder = "test"
    val_folder = "val"
    def __init__(
            self,
            root_dir: str="./nerf_synthetic/lego/",
            split: str = "train",
            scaled_size : tuple[int, int] = (100, 100)
    ):
        self.root = os.path.expanduser(root_dir)
       
        self.split = split
        json_path = os.path.join(root_dir, f"transforms_{self.split}.json")
        self.tform_dataset, self.img_paths_dataset = self.load_tforms_and_file_path(json_path)
        

        self.h_og = 800 
        self.w_og = 800
        self.new_size = scaled_size

        

        img_path = os.path.normpath(os.path.join(self.root, f'{self.img_paths_dataset[0]}.png'))
        print(img_path)
        img = self.load_img_from_path(img_path)
        img = np.array(img)
        
        self.h, self.w, _ = img.shape

        self.cam_intrinsics = self.get_intrinsics()

        

    def __len__(self) -> int:
        return len(self.img_paths_dataset)
    
    def __getitem__(self, i: int) -> Tuple:
        img = None
        class_idx = None
        img_path = os.path.normpath(os.path.join(self.root, f'{self.img_paths_dataset[i]}.png'))
        img = self.load_img_from_path(img_path)
        img_tensor = torchvision.transforms.ToTensor()(img).permute(1,2,0)

        img = np.array(img)
        cam_2world = self.tform_dataset[i]

        # return img, cam_2world, self.cam_intrinsics
        return img_tensor, cam_2world
        

    def get_intrinsics(self) -> torch.Tensor:
        scale = (self.new_size[0]/800)

        intrinsics = torch.eye(3,3)
        focal = self.w_og / (2 * np.tan(self.camera_angle_x / 2))
        focal *= scale
        intrinsics[0, 0] = focal
        intrinsics[1, 1] = focal
        intrinsics[0, -1] = (self.w_og/ 2 ) * scale
        intrinsics[1, -1] = (self.h_og / 2 ) * scale
        return intrinsics

    def load_img_from_path(self, path: str) -> Image:
        img = Image.open(path).convert('RGB')
        
        resized_img = img.resize(self.new_size, Image.Resampling.LANCZOS)
        # resized_img = img.resize(new_size, Image.LANCZOS)
        return resized_img
    
    def load_tforms_and_file_path(self, json_path: str) -> Tuple[torch.Tensor, List[str]]:
        
        tforms_list = []
        path_list = []
        with open(json_path) as f:
            data = json.load(f)
            self.camera_angle_x = data["camera_angle_x"]
            for frame in data["frames"]:
                path_list.append(frame['file_path'])
                # tforms_list.append(torch.tensor(frame["transform_matrix"]))
                tform = torch.tensor(frame["transform_matrix"])
                tform = tform.clone()
                tform[:3, 1:3] *= -1  # Flip Y and Z axes (OpenGL -> OpenCV)
                tform[:3, 3] /= 3.0
                tforms_list.append(tform)
        
        tforms = torch.stack(tforms_list, dim=0)
        return tforms, path_list

    def get_hw(self) -> Tuple[int, int]:
        return self.h, self.w


    