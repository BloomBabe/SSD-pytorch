import torch
import numpy as np 
import torch.nn as nn
from utils.modules.backbone import vgg_backbone
from utils.modules.l2norm import L2Norm
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

    def forward(self, x):
        out = []
        for layer in self.layers_block:
            x = layer(x)
            out.append(x)
        return out

class MultiBox(nn.Module):
    """ Classifier and box regressor heads"""
    def __init__(self,
                 cfg,
                 num_classes):
        super(MultiBox, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = cfg['num_boxes']
        self.input_channels = cfg['in_channels']
        assert len(self.num_boxes) == len(self.input_channels)

        self.loc_layers = nn.ModuleList(self._make_locreg())
        self.cls_layers = nn.ModuleList(self._make_cls())

    def _make_cls(self):
        """Make classifier heads"""
        layers: List[nn.Module] = []
        for num_box, in_channels in zip(self.num_boxes, self.input_channels):
            layers += [nn.Conv2d(in_channels, self.num_classes*num_box, kernel_size=3, padding=1)]
        return layers

    def _make_locreg(self):
        """Make location regressor heads"""
        layers: List[nn.Module] = []
        for num_box, in_channels in zip(self.num_boxes, self.input_channels):
            layers += [nn.Conv2d(in_channels, 4*num_box, kernel_size=3, padding=1)]
        return layers

    def forward(self, list_x):
        assert (len(list_x) == len(self.loc_layers)) and(len(list_x) == len(self.cls_layers))
        cls_logits = list() 
        loc_logits = list()
        for k, cls_layer, loc_layer in enumerate(zip(self.cls_layers, self.loc_layers)):
            assert list_x[k] == self.input_channels[k]
            cls_logits.append(cls_layer(list_x[k]))
            loc_logits.append(loc_layer(list_x[k]))
        return cls_logits, loc_logits

class SSD(nn.Module):
    """ SSD model """
    def __init__(self,
                 num_classes = 100,
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
        self.multi_box = MultiBox(self.multibox_cfg, self.num_classes)

    def forward(self, x):
        sources = list()

        conv4, out = self.backbone(x)
        conv4 = self.l2norm(conv4)
        sources.append(conv4)
        outs = self.ssd_layers(out)
        for out in outs:
            sources.append(out)

        cls_outs, loc_outs = self.multi_box(sources)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc_outs], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in cls_outs], 1)

        if self.mode == 'train':
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))
        return output
          