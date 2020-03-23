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
from .base_model import BaseModel
from . import networks
import time


class CssGANModel(BaseModel):
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
        parser.set_defaults(norm='batch',netG='unet_256')
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
        self.loss_names = ['G_GAN','G_classifier','G_total','D_real','D_fake','D_classifier_real','D_total']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real_A', 'real_B', 'fake_B']
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
        if self.isTrain:
            self.netD = self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,classifier_nc=opt.dataset_num)
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
        self.fake_B = self.netG(x)  # generate output image given the input data_A

    def backward_G(self):
        fake_AB = torch.cat((self.real_A,self.fake_B),1)
        pred_fakes,cls = self.D_output(self.netD(fake_AB))
        self.loss_G_GAN = 0
        for pred_fake in pred_fakes:
            self.loss_G_GAN += self.criterionGAN(pred_fake,True)
        self.loss_G_GAN /= len(pred_fakes)
        self.loss_G_L1 = self.criterionL1(self.fake_B,self.real_B) * self.opt.lambda_L1

        self.loss_G_classifier = self.classifier_loss(cls,self.tag)

        self.loss_G_total = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_classifier
        self.loss_G_total.backward()

    def D_output(self,outputs):
        tmp = list(outputs)
        classifier_loss = tmp[-1]
        tmp.pop()
        return tmp,classifier_loss

    def backward_D(self):
        # Fake
        fake_AB = torch.cat((self.real_A,self.fake_B),1)
        pred_fakes, _ = self.D_output(self.netD(fake_AB.detach()))
        self.loss_D_fake = 0
        for pred_fake in pred_fakes:
            self.loss_D_fake += self.criterionGAN(pred_fake,False)
        self.loss_D_fake /= len(pred_fakes)
        # Real
        real_AB = torch.cat((self.real_A,self.real_B),1)
        pred_reals,cls_real = self.D_output(self.netD(real_AB.detach()))
        self.loss_D_real = 0
        for pred_real in pred_reals:
            self.loss_D_real += self.criterionGAN(pred_real,True)
        self.loss_D_real /= len(pred_reals)

        if self.opt.debug:
            print(cls_real.dtype,self.tag.dtype)
            print(cls_real.size(),self.tag.size())

        self.loss_D_classifier_real = self.classifier_loss(cls_real,self.tag)
        self.loss_D_total = self.loss_D_fake + self.loss_D_real + self.loss_D_classifier_real
        self.loss_D_total.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results

        # update D
        self.set_requires_grad(self.netD,True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # updateG
        self.set_requires_grad(self.netD,False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_net(self):
        return self.netG,self.netD
