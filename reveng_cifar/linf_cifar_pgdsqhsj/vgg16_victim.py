#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:05:56 2021

@author: xiawei
"""


import torch.nn as nn
import torch
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG_plain(nn.Module):
    def __init__(self, vgg_name, nclass, img_width=32):
        super(VGG_plain, self).__init__()
        self.img_width = img_width
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)

    def get_feature(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

def vgg16():
    return VGG_plain('VGG16', nclass=10)

#model = vgg16()
#model.load_state_dict(torch.load('./models/cifar10/vgg16.pth'))
#state_dict1 = torch.load('./models/cifar10/vgg16.pth')
#model.eval()
#a  = np.load('./features/cifar10/CW_adv.npy')
#model(torch.tensor(a[:10].reshape([-1,3,32,32])))
#model.get_feature(torch.tensor(a[:10].reshape([-1,3,32,32])))

# victim.get_feature(x_hat)