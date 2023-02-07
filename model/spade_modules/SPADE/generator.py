"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork
from .architecture import SPADEResnetBlock as SPADEResnetBlock

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, input_nc, output_nc, num_upsampling_layers='normal', crop_size=256, aspect_ratio=1):
        super().__init__()
        # self.opt = opt
        nf = 64
        norm_G = 'spectralinstance'
        label_nc = 2
        self.num_upsampling_layers = num_upsampling_layers
        self.sw, self.sh = self.compute_latent_vector_size(num_upsampling_layers, crop_size, aspect_ratio)
        norm_layer = nn.InstanceNorm2d
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, nf, kernel_size=7, padding=0),
                 norm_layer(nf),
                 nn.ReLU(True)]
            
        # encoder
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(nf * mult, nf* mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(nf * mult * 2),
                      nn.ReLU(True)]
        self.model = nn.Sequential(*model)
            
        # bottom
        mult = 2 ** n_downsampling
        self.G_middle_0 = SPADEResnetBlock(mult * nf, mult * nf, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(mult * nf, mult * nf, norm_G, label_nc)

        # decoder
        self.up = nn.Upsample(scale_factor=2)
        mult = 2 ** (n_downsampling - 0)
        self.up_0 = SPADEResnetBlock(mult * nf, int(nf * mult / 2), norm_G, label_nc)
        mult = 2 ** (n_downsampling - 1)
        self.up_1 = SPADEResnetBlock(mult * nf, int(nf * mult / 2), norm_G, label_nc)

        if num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, norm_G, label_nc)

        # self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        conv_img = [nn.ReflectionPad2d(3),
                      nn.Conv2d(nf, output_nc, kernel_size=7, padding=0)]
        self.conv_img = nn.Sequential(*conv_img)

    def compute_latent_vector_size(self, num_upsampling_layers, crop_size, aspect_ratio):
        if num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif num_upsampling_layers == 'more':
            num_up_layers = 6
        elif num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('num_upsampling_layers [%s] not recognized' %
                             num_upsampling_layers)

        sw = crop_size // (2**num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh

    def forward(self, input, seg=None):
        x = self.model(input)
        x = self.G_middle_0(x, seg)

        if self.num_upsampling_layers == 'more' or \
           self.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x