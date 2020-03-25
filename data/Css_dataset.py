"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params,get_transform
from PIL import Image
import os
import random
import time

class CssDataset(BaseDataset):
    """A  dataset class can load CssGan DataSet."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self,opt)
        self.dir_A = os.path.join(opt.dataroot,opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot,opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A,opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        #time0 = time.time()

        index = index % self.A_size
        A_path = self.A_paths[index]
        dir_name,file_name = os.path.split(A_path)
        B_path = os.path.join(self.dir_B,file_name)

        #time2 = time.time()

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        #time3 = time.time()

        transform_params = get_params(self.opt,A_img.size)
        A_transform = get_transform(self.opt,transform_params,grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt,transform_params,grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        #time4 = time.time()

        #time1 = time.time()
        #print('total time: ',time1-time0,'; path: ',time2-time0,';open: ',time3-time2,';transform:',time4-time3)

        return {'A':A,'B':B,'A_paths':A_path,'B_paths':B_path}

    def __len__(self):
        """Return the total number of images."""
        return self.A_size
