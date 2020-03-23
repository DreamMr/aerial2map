import os.path
from data.base_dataset import BaseDataset, get_params,get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
import random
import torch

class MapUnalignedDataset(BaseDataset):
    """
    This dataset class can load my map dataset
    """

    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        self.dir_A = os.path.join(opt.dataroot,opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot,opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A,opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = self.A_size
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc

    def __getitem__(self, index):
        index = index % self.A_size
        A_path = self.A_paths[index]
        prefix_B = 'tw_16_e_all_'
        (_,file_name) = os.path.split(A_path)
        file,_ = os.path.splitext(file_name)
        file_B_name = prefix_B + file.split('_')[-1] + '.tif'
        B_path = os.path.join(self.opt.dataroot,self.opt.phase + 'B',file_B_name)


        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        return self.A_size
