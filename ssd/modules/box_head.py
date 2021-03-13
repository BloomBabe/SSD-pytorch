import torch
import math
import numpy as np 
import torch.nn as nn
from itertools import product as product


class SSDMultiBox(nn.Module):
    """ Classifier and box regressor heads"""
    def __init__(self,
                 cfg,
                 num_classes,
                 device):
        super(SSDMultiBox, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = cfg['num_boxes']
        self.input_channels = cfg['in_channels']
        self.fmap_wh = cfg['fmap_wh']
        self.aspect_ratios = cfg['aspect_ratios']
        self.scales = cfg['scales']
        self.device = device
        assert len(self.num_boxes) == len(self.input_channels)

        self.reg_layers = nn.ModuleList(self._make_locreg())
        self.cls_layers = nn.ModuleList(self._make_cls())
        self._init_weights()
        self.default_boxes = self._init_default_boxes()

    def _make_cls(self):
        """Make classifier head"""
        layers: List[nn.Module] = []
        for num_box, in_channels in zip(self.num_boxes, self.input_channels):
            layers += [nn.Conv2d(in_channels, self.num_classes*num_box, kernel_size=3, padding=1)]
        return layers

    def _make_locreg(self):
        """Make location regressor head"""
        layers: List[nn.Module] = []
        for num_box, in_channels in zip(self.num_boxes, self.input_channels):
            layers += [nn.Conv2d(in_channels, 4*num_box, kernel_size=3, padding=1)]
        return layers

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def _init_default_boxes(self):
        assert len(self.aspect_ratios)==len(self.fmap_wh)
        assert len(self.aspect_ratios)==len(self.scales)
        default_boxes = []
        for k in range(len(self.fmap_wh)):
            for i, j in product(range(self.fmap_wh[k]), repeat=2):
                cx = (j + 0.5) / self.fmap_wh[k]
                cy = (i + 0.5) / self.fmap_wh[k]
                for ratio in self.aspect_ratios[k]:
                    #(cx, cy, w, h)
                    default_boxes.append([cx, cy, self.scales[k]*math.sqrt(ratio), 
                                          self.scales[k]/math.sqrt(ratio)]) 
                    if ratio == 1:
                        try:
                            add_scale = math.sqrt(self.scales[k]*self.scales[k+1])
                        except IndexError:
                            #for the last feature map
                            add_scale = 1.
                        default_boxes.append([cx, cy, add_scale, add_scale])
        default_boxes = torch.FloatTensor(default_boxes).to(self.device)
        default_boxes.clamp_(0, 1)
        return default_boxes

    def forward(self, list_x):
        assert (len(list_x) == len(self.reg_layers)) and(len(list_x) == len(self.cls_layers))
        cls_logits = list() 
        box_preds = list()
        for k, (cls_layer, loc_layer) in enumerate(zip(self.cls_layers, self.reg_layers)):
            assert list_x[k].size(1) == self.input_channels[k]
            cls_logits.append(cls_layer(list_x[k]).permute(0, 2, 3, 1).contiguous())
            box_preds.append(loc_layer(list_x[k]).permute(0, 2, 3, 1).contiguous())

        batch_size = list_x[0].size(0)  
        cls_logits = torch.cat([c.view(c.size(0), -1) for c in cls_logits], dim=1).view(batch_size, -1, self.num_classes)
        box_preds = torch.cat([l.view(l.size(0), -1) for l in box_preds], dim=1).view(batch_size, -1, 4)
        return cls_logits, box_preds