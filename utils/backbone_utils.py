import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import numpy as np 
import json
import os


def vgg_backbone(cfg,
                 input_channels = 3,
                 pretrained = True):
    if (input_channels != 3) and pretrained:
        raise ValueError('There are available weights only for 3 channels models.')
    model = VGGBackbone(cfg=cfg, input_channels=input_channels,
                        pretrained=pretrained)
    return model

class VGGBackbone(nn.Module):
    """ Build VGG model without classifier"""
    def __init__(self,
                 cfg,
                 input_channels = 3,
                 pretrained = True):
        super(VGGBackbone, self).__init__()
        self.cfg = cfg 
        self.input_channels = input_channels
        self.batch_norm = bool(self.cfg['batch_norm'])
        self.url = str(self.cfg['url'])
        self.build = self.cfg['cfg']
        self.pretrained = pretrained

        self.features = self._make_layers()
        self._init_weights()

    def _make_layers(self):
        # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
        layers: List[nn.Module] = []
        in_channels = self.input_channels
        for v in self.build:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _init_weights(self):
        if self.pretrained:
            state_dict = load_state_dict_from_url(self.url,
                                                  progress=False)
            self.load_state_dict(state_dict, strict=False)
        else:    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        num_layer = 0
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                num_layer +=1
            if num_layer == 4:
                conv4_x = x.detach().clone()
            x = layer(x)
        return conv4_x, x