import torch
import numpy as np 
import torch.nn as nn
from utils.backbone_utils import vgg_backbone
from utils.l2norm import L2Norm
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SSDLayers(nn.Module):
    """ SSD Extra-convolutional layers : conv6 - conv11 in paper"""
    def __init__(self,
                 cfg,
                 input_channels = 512):
        super(SSDLayers, self).__init__()
        self.cfg = cfg
        self.input_channels = input_channels
        self.layers_block = self._make_layers()
        
    def _make_layers(self):
        blocks = nn.ModuleList()
        layers: List[nn.Module] = []
        in_channels = self.input_channels
        for k, v in enumerate(self.cfg):
            if v == "M":
                params = self.cfg[k+1]
                layers += [nn.MaxPool2d(kernel_size=params[0], stride=params[1], padding=params[2])]
            elif v == "Conv":
                params = self.cfg[k+1]
                layers += [nn.Conv2d(in_channels, params[0], kernel_size=params[1], padding=params[2], stride=params[3], dilation=params[4]),
                            nn.ReLU(inplace=True)]
                in_channels = params[0]
            elif v == "outConv":
                params = self.cfg[k+1]
                layers += [nn.Conv2d(in_channels, params[0], kernel_size=params[1], padding=params[2], stride=params[3], dilation=params[4]),
                           nn.ReLU(inplace=True)]
                in_channels = params[0]
                blocks.append(nn.Sequential(*layers))
                layers = []
        return blocks

    def forward(self, x):
        out = []
        for layer in self.layers_block:
            x = layer(x)
            out.append(x)
        return out


class SSD(nn.Module):
    """ SSD model """
    def __init__(self,
                 num_classes = 100,
                 backbone_name = 'vgg16_bn',
                 ssd_layers_name = 'ssd_300'):
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.ssd_layers_name = ssd_layers_name

        assert self.backbone_name in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
                                      'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        assert self.ssd_layers_name in ['ssd_300']

        self.file_pth = os.path.join(BASE_DIR,'utils/architecture_cfg.json')
        with open(self.file_pth) as f:
            cfgs = json.load(f)
        self.backbone_cfg = cfgs['backbone']
        self.ssd_layers_cfg = cfgs['ssd_layers']

        self.backbone = vgg_backbone(self.backbone_cfg[backbone_name], input_channels=3, pretrained=True)
        self.l2norm = L2Norm(512, 20)
        self.ssd_layers = SSDLayers(self.ssd_layers_cfg[ssd_layers_name])  

    def forward(self, x):
        conv4, out = self.backbone(x)
        conv4 = self.l2norm(conv4)
        out_2, out_3, out_4, out_5 = self.ssd_layers(out)
        return out_1, out_2, out_3, out_4, out_5
          