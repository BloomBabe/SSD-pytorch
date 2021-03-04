import torch
import numpy as np 
import torch.nn as nn
from utils.modules.backbone import vgg_backbone
from utils.modules.l2norm import L2Norm
from utils.modules.box_head import SSDMultiBox
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SSDLayers(nn.Module):
    """ SSD Extra-convolutional layers : conv6 - conv11 in paper"""
    def __init__(self,
                 cfg,
                 in_channels = 512):
        super(SSDLayers, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.layers_block = self._make_layers()
        self._init_weights()

    def _make_layers(self):
        blocks = nn.ModuleList()
        layers: List[nn.Module] = []
        input_channels = self.in_channels
        for k, v in enumerate(self.cfg):
            if v == "M":
                params = self.cfg[k+1]
                layers += [nn.MaxPool2d(kernel_size=params[0], stride=params[1], padding=params[2])]
            elif v == "Conv":
                params = self.cfg[k+1]
                layers += [nn.Conv2d(input_channels, params[0], kernel_size=params[1], padding=params[2], stride=params[3], dilation=params[4]),
                            nn.ReLU(inplace=True)]
                input_channels = params[0]
            elif v == "outConv":
                params = self.cfg[k+1]
                layers += [nn.Conv2d(input_channels, params[0], kernel_size=params[1], padding=params[2], stride=params[3], dilation=params[4]),
                           nn.ReLU(inplace=True)]
                input_channels = params[0]
                blocks.append(nn.Sequential(*layers))
                layers = []
        return blocks

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = []
        for layer in self.layers_block:
            x = layer(x)
            out.append(x)
        return out


class SSD(nn.Module):
    """ SSD model """
    def __init__(self,
                 num_classes = 101,
                 mode = 'train',
                 backbone_name = 'vgg16_bn',
                 ssd_layers_name = 'ssd_300'):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.backbone_name = backbone_name
        self.ssd_layers_name = ssd_layers_name

        assert mode in ['train', 'test']
        assert self.backbone_name in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
                                      'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        assert self.ssd_layers_name in ['ssd_300']

        self.file_pth = os.path.join(BASE_DIR,'utils/architecture_cfg.json')
        with open(self.file_pth) as f:
            cfgs = json.load(f)
        self.backbone_cfg = cfgs['backbone']
        self.ssd_layers_cfg = cfgs['ssd_layers']
        self.multibox_cfg = cfgs['multi_box']

        self.backbone = vgg_backbone(self.backbone_cfg[backbone_name], in_channels=3, pretrained=True)
        self.l2norm = L2Norm(512, 20)
        self.ssd_layers = SSDLayers(self.ssd_layers_cfg[ssd_layers_name], in_channels=512)  
        self.multi_box = SSDMultiBox(self.multibox_cfg, self.num_classes)

    def forward(self, x):
        sources = list()

        conv4, out = self.backbone(x)
        conv4 = self.l2norm(conv4)
        sources.append(conv4)
        outs = self.ssd_layers(out)
        for out in outs:
            sources.append(out)

        cls_logits, box_preds = self.multi_box(sources)
        return cls_logits, box_preds
          