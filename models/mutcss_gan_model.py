"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import numpy as np
from .base_model import BaseModel
from . import networks
import time


class MutCssGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='Css')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        parser.set_defaults(norm='batch',netG='mut_unet',netD='mut_css')
        if is_train:
            parser.set_defaults(pool_size=0,gan_mode='vanilla')
            parser.add_argument('--lambda_L1',type=float,default=100.0,help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN','G_classifier','G_total','D_real','D_fake','D_classifier_real','D_total','G_L1']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real_A', 'real_B', 'fake_B_high','fake_B_middle','fake_B_low']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G','D']
        else:
            self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc + opt.dataset_num, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = self.netG.cuda(self.device)
        self.G_output_size = [256,128,64]
        self.downsample = torch.nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        if self.isTrain:
            self.netD = self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                      self.gpu_ids, classifier_nc=opt.dataset_num,D_num=[3,2,1])
            self.netD = self.netD.cuda(self.device)
        if self.isTrain:  # only defined during training time
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #self.classifier_loss =torch.nn.CrossEntropyLoss()



        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks
    def classifier_loss(self,logit,target):
        return torch.nn.functional.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.tag = input['tag'].to(self.device)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        tag = self.tag.view(self.tag.size(0),self.tag.size(1),1,1).clone()
        c = tag.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
        x = torch.cat([self.real_A,c],dim=1)

        self.fake_B_high,self.fake_B_middle,self.fake_B_low = self.netG(x)  # generate output image given the input data_A
        self.real_A_high = self.real_A
        self.real_A_middle = self.downsample(self.real_A_high)
        self.real_A_low = self.downsample(self.real_A_middle)

        self.real_B_high = self.real_B
        self.real_B_middle = self.downsample(self.real_B_high)
        self.real_B_low = self.downsample(self.real_B_middle)

    def backward_G(self):
        self.loss_G_GAN = 0
        self.loss_G_L1 = 0
        self.loss_G_classifier = 0


        fake_ABs = [torch.cat((self.real_A_high,self.fake_B_high),1).to(self.device),
                    torch.cat((self.real_A_middle,self.fake_B_middle),1).to(self.device),
                    torch.cat((self.real_A_low,self.fake_B_low),1).to(self.device)]

        pred_fakes,clss = self.netD(fake_ABs)
        for i in range(len(pred_fakes)):
            pred_fake = pred_fakes[i]
            for pred in pred_fake:
                self.loss_G_GAN += self.criterionGAN(pred,True)
            self.loss_G_classifier += self.classifier_loss(clss[i],self.tag)
        self.loss_G_L1 = (self.criterionL1(self.fake_B_high,self.real_B_high) + self.criterionL1(self.fake_B_middle,self.real_B_middle) + self.criterionL1(self.fake_B_low,self.real_B_low)) * self.opt.lambda_L1
        self.loss_G_total = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_classifier
        self.loss_G_total.backward()

    def D_output(self,outputs):
        tmp = list(outputs)
        classifier_loss = tmp[-1]
        tmp.pop()
        return tmp,classifier_loss

    def backward_D(self):
        self.loss_D_fake = 0
        self.loss_D_real = 0
        self.loss_D_classifier_real = 0
        # Fake
        fake_ABs = [torch.cat((self.real_A_high, self.fake_B_high), 1).to(self.device),torch.cat((self.real_A_middle, self.fake_B_middle), 1).to(self.device),torch.cat((self.real_A_low, self.fake_B_low), 1).to(self.device)]
        pred_fakes,_ = self.netD(fake_ABs)

        for pred_fake in pred_fakes:
            for pred in pred_fake:
                self.loss_D_fake += self.criterionGAN(pred,False)
        # Real
        realAB = [torch.cat((self.real_A_high, self.real_B_high), 1).to(self.device),
                    torch.cat((self.real_A_middle, self.real_B_middle), 1).to(self.device),
                    torch.cat((self.real_A_low, self.real_B_low), 1).to(self.device)]
        pred_reals,clss = self.netD(realAB)
        for i in range(len(pred_reals)):
            pred_real = pred_reals[i]
            for pred in pred_real:
                self.loss_D_real += self.criterionGAN(pred,True)
            self.loss_D_classifier_real += self.classifier_loss(clss[i],self.tag)

        self.loss_D_total = self.loss_D_fake + self.loss_D_real + self.loss_D_classifier_real
        self.loss_D_total.backward()



    def optimize_parameters(self):
        #time0 = time.time()
        """Update network weights; it will be called in every training iteration."""
        #time1 = time.time()
        self.forward()               # first call forward to calculate intermediate results
        #time2 = time.time()

        # update D
        self.set_requires_grad(self.netD,True)
        self.optimizer_D.zero_grad()
        #time3 = time.time()
        self.backward_D()
        #time4 = time.time()
        self.optimizer_D.step()

        # updateG
        self.set_requires_grad(self.netD,False)
        self.optimizer_G.zero_grad()
        #time5 = time.time()
        self.backward_G()
        #time6 = time.time()
        self.optimizer_G.step()
        #time7 = time.time()

        #print('forward: ',time2-time1,';optiD: ',time4-time3,';optiG: ',time6-time5,';total time: ',time7-time0)

    def get_net(self):
        return self.netG,self.netD
