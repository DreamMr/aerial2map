import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import numpy as np
import cv2
import torch

def get_img(visuals,A_path,B_path,opt,epoch_name,dataset_name):
    fake_B = get_bgr(util.tensor2im(visuals['fake_B']))
    b_name = B_path.split('\\')[-1]
    path = './' + opt.test_out_name + '/' + dataset_name + '/' +str(epoch_name) + "/" + b_name
    cv2.imwrite(path,fake_B)

def get_bgr(img):
    b,g,r = cv2.split(img)
    result = cv2.merge([r,g,b])
    return result

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_filp = True
    opt.display_id = -1

    epoch_list = [40,45,50,55,60,65,70,75]

    dataset_names = {'maps':'./datasets/maps/','my_map':'./datasets/my_map'}
    for epoch_name in epoch_list:
        opt.epoch = epoch_name
        dataset_name_list = list(dataset_names.keys())
        for t in range(len(dataset_name_list)):
            dataset_key = dataset_name_list[t]
            opt.dataroot = dataset_names[dataset_key]
            dataset = create_dataset(opt)
            model = create_model(opt)
            model.setup(opt)

            if opt.eval:
                model.eval()
            tag = np.zeros(shape=[1,len(dataset_name_list)])
            tag[0][t] = 1.0
            tag = torch.from_numpy(tag).float()
            for i, data in enumerate(dataset):
                if i >= opt.num_test:
                    break
                data['tag'] = tag
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                get_img(visuals,data['A_paths'][0],data['B_paths'][0],opt,epoch_name,dataset_key)

