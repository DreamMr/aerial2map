from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util import util
from tensorboardX import SummaryWriter
import numpy as np
import cv2
import os
import torch
import random
import json

import time

if __name__ == '__main__':
    opt = TrainOptions().parse()

    dataset_dic = {'14':'./datasets/shanghai/14','15':'./datasets/shanghai/15','16':'./datasets/shanghai/16','17':'./datasets/shanghai/17','18':'./datasets/shanghai/18'}

    dataset_keys = list(dataset_dic.keys())
    dataset_list = {}
    dataset_size_list = []
    dataset_tag_list = {}
    for i in range(len(dataset_keys)):
        dataset_name = dataset_keys[i]
        opt.dataroot = dataset_dic[dataset_name]
        dataset = create_dataset(opt)
        dataset_size = len(dataset)
        dataset_list[dataset_name] = dataset
        dataset_size_list.append(dataset_size)

        tag = np.zeros(shape=[len(dataset_keys)])
        tag[i] = 1.0
        dataset_tag_list[dataset_name] = tag
    sz = min(dataset_size_list) //opt.batch_size
    opt.dataset_num = len(dataset_keys)
    if opt.debug:
        print(dataset_tag_list)
        print(dataset_list)
    # # load l_map
    # opt.dataroot = './datasets/maps'
    # l_map_dataset = create_dataset(opt)
    # l_map_dataset_size = len(l_map_dataset)
    # # load map_16
    # opt.dataroot = './datasets/my_map'
    # my_map_dataset = create_dataset(opt)
    # my_map_dataset_size = len(my_map_dataset)
    # # load map_17
    # opt.dataroot = './datasets/map_17'
    # map_17_dataset = create_dataset(opt)
    # map_17_dataset_size = len(map_17_dataset)

    model = create_model(opt) # css_gan
    model.setup(opt)

    writer = SummaryWriter()
    #sample_input_G = torch.rand(1,3+opt.dataset_num,256,256)
    #sample_input_D = torch.rand(1,6,256,256)
    #netG,netD = model.get_net()
    #writer.add_graph(netG,(sample_input_G,))
    #writer.add_graph(netD,(sample_input_D,))

    total_iters = opt.epoch_count * opt.batch_size
    if opt.debug:
        print("size: ",sz)
    for epoch in range(opt.epoch_count,opt.niter + opt.niter_decay + 1):
        epoch_iter = 0
        each_iter = 0
        for i in range(sz):
            dataset_keys = list(dataset_dic.keys())
            random.shuffle(dataset_keys)
            if opt.debug:
                print(dataset_keys)
            for dataset_name in dataset_keys:
                dataset = None
                tag = None
                dataset = dataset_list[dataset_name]
                a_tag = dataset_tag_list[dataset_name]
                t = np.expand_dims(a_tag,axis=0).repeat(opt.batch_size,axis=0)
                tag = torch.from_numpy(t).float()


                data = next(iter(dataset))
                data['tag'] = tag

                model.set_input(data)
                model.optimize_parameters()
                losses = model.get_current_losses()
                loss_json = json.dumps(losses)

                if total_iters % opt.print_freq:
                    writer.add_text(dataset_name+'/','epoch: ' + str(epoch) +' epoch_iter:'+ str(epoch_iter)+ ' each_iter:'+str(each_iter) +' loss: ' + loss_json,epoch)
                    print('epoch : ',epoch,' iter:',epoch_iter,' each_iter: ',each_iter,' loss: ',loss_json)

                if total_iters % opt.display_freq:
                    visuals = model.get_current_visuals()
                    real_A = util.tensor2im(visuals['real_A'])
                    fake_B = util.tensor2im(visuals['fake_B_high'])
                    real_B = util.tensor2im(visuals['real_B'])

                    output = np.concatenate((real_A,fake_B,real_B),axis=1)
                    b,g,r = cv2.split(output)
                    result = cv2.merge([r,g,b])
                    result = torch.from_numpy(result)
                    writer.add_image(dataset_name+'/result',result,total_iters,dataformats='HWC')

                    fake_feature = {'high':util.tensor2im(visuals['fake_B_high']),'middle':util.tensor2im(visuals['fake_B_middle']),'low':util.tensor2im(visuals['fake_B_low'])}
                    keys = ['high','middle','low']
                    for key in keys:
                        writer.add_image(dataset_name+'/feature/'+key,fake_feature[key],total_iters,dataformats='HWC')

                    writer.add_scalar(dataset_name+'/G_GAN',losses['G_GAN'],total_iters)
                    writer.add_scalar(dataset_name+'/G_classifier',losses['G_classifier'],total_iters)
                    writer.add_scalar(dataset_name + '/G_total',losses['G_total'],total_iters)
                    writer.add_scalar(dataset_name + '/D_real',losses['D_real'],total_iters)
                    writer.add_scalar(dataset_name + '/D_fake',losses['D_fake'],total_iters)
                    writer.add_scalar(dataset_name + '/D_classifier_real',losses['D_classifier_real'],total_iters)
                    writer.add_scalar(dataset_name + '/D_total',losses['D_total'],total_iters)

                epoch_iter += opt.batch_size
                total_iters += opt.batch_size
            each_iter += opt.batch_size

        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()

    writer.close()
