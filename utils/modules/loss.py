import torch.nn as nn 
from utils.box_utils import *


class MultiBoxLoss(nn.Module):
    """
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    """
    def __init__(self,
                 default_boxes,
                 device,
                 threshold = 0.5,
                 neg_pos = 3,
                 alpha = 1.):
        super(MultiBoxLoss, self).__init__()
        self.default_boxes = default_boxes
        self.device = device
        self.threshold = threshold
        self.neg_pos = neg_pos
        self.alpha = alpha

    def forward(self, loc_pred, cls_pred, gt_boxes, gt_labels):
        """
            Forward propagation
            loc_pred: Pred location, a tensor of dimensions (N, 8732, 4)
            cls_pred:  Pred class scores for each of the encoded boxes, a tensor fo dimensions (N, 8732, n_classes)
            boxes: True object bouding boxes, a list of N tensors
            labels: True object labels, a list of N tensors
            
            Out: Mutilbox loss
        """
        batch_size = loc_pred.size(0)
        num_classes = cls_pred.size(2)
        default_boxes_xy = cxcy_to_xy(default_boxes)
        for i in range(batch_size):
            n_objects = gt_boxes[i].size(0)
            overlaps = compute_iou(gt_boxes[i], default_boxes_xy) # iou size:(num_gt_boxes, num_default_boxes)
            overlaps_per_default_box, objects_per_default_box = overlaps.max(dim=0) 
            _, defaults_per_gt_boxes = overlaps.max(dim=1)
            objects_per_default_box[defaults_per_gt_boxes] = torch.LongTensor(range(n_objects)).to(self.device)
            overlaps_per_default_box[defaults_per_gt_boxes] = 1.


