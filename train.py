"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util import util
from tensorboardX import SummaryWriter
import numpy as np
import cv2
import os
import torch
import json
import time
import torchvision
train_out = './train_out/'

def save_img(visuals,epoch,index,A_path,B_path,opt,writer):
    if opt.model != 'pix2pix':
        A_file_name = os.path.basename(A_path).split('.')[0].split('_')[-1]
        B_file_name = os.path.basename(B_path).split('.')[0].split('_')[-1]
        img_name = './' + opt.train_out + '/' + str(epoch) + '_' + str(index) + '_' + A_file_name + '_' + B_file_name + '.jpg'
        real_A = util.tensor2im(visuals['real_A'])
        # real_A = visuals['real_A'].view([256,256,3]).cpu().float().numpy()
        fake_B = util.tensor2im(visuals['fake_B'])
        rec_A = util.tensor2im(visuals['rec_A'])
        idt_A = util.tensor2im(visuals['idt_A'])

        real_B = util.tensor2im(visuals['real_B'])
        #  real_B = visuals['real_B'].view([256,256,3]).cpu().float().numpy()
        fake_A = util.tensor2im(visuals['fake_A'])
        rec_B = util.tensor2im(visuals['rec_B'])
        idt_B = util.tensor2im(visuals['idt_B'])

        up = np.concatenate((real_A,fake_B,rec_A,idt_B),axis=1)
        bottom = np.concatenate((real_B,fake_A,rec_B,idt_A),axis=1)
        output = np.concatenate((up,bottom),axis=0)

        b,g,r = cv2.split(output)
        result = cv2.merge([r,g,b])
        #cv2.imwrite(img_name,result)
        writer.add_image('train_image',result,epoch)
    else:
        A_file_name = os.path.basename(A_path).split('.')[0].split('_')[-1]
        B_file_name = os.path.basename(B_path).split('.')[0].split('_')[-1]
        img_name = './' + opt.train_out + '/' + str(epoch) + '_' + str(
            index) + '_' + A_file_name + '_' + B_file_name + '.jpg'
        real_A = util.tensor2im(visuals['real_A'])
        # real_A = visuals['real_A'].view([256,256,3]).cpu().float().numpy()
        fake_B = util.tensor2im(visuals['fake_B'])

        real_B = util.tensor2im(visuals['real_B'])
        #  real_B = visuals['real_B'].view([256,256,3]).cpu().float().numpy()
        output = np.concatenate((real_A, fake_B, real_B), axis=1)

        b, g, r = cv2.split(output)
        result = cv2.merge([r, g, b])
        #cv2.imwrite(img_name, result)
        result = torch.from_numpy(result)
        print(type(result))
        print(result.shape)

        print(result)
        writer.add_image('train_image', result,epoch,dataformats='HWC')
        out = result.numpy()
        cv2.imwrite(img_name, out)


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    count = 0
    writer = SummaryWriter()
    #input = torch.rand(13,1,28,28)
    #writer.add_graph(model,(input,))
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            time1 = time.time()
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            time2 = time.time()
            print(time2-time1)
            losses = model.get_current_losses()
            loss_json = json.dumps(losses)
            print('epoch: ',epoch,'   index: ',i,'  loss: ', loss_json)
            writer.add_text('text/loss','epoch: '+ str(epoch_iter) + ' loss: ' + loss_json,epoch_iter)
            if count % opt.save_imgs == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                save_img(model.get_current_visuals(),epoch,i,data['A_paths'][0],data['B_paths'][0],opt,writer)
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if count % opt.save_loss == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                print('loss: ',losses)
                for label, val in losses.items():
                    writer.add_scalar(label, val, count)

                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                '''
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    '''

            if i % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            count += 1

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)



        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    writer.close()