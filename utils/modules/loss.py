import torch.nn as nn 
import torch.nn.functional as F
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
            gt_boxes: True object bouding boxes, a list of N tensors
            gt_labels: True object labels, a list of N tensors
            
            Out: Mutilbox loss
        """
        batch_size = loc_pred.size(0)
        num_classes = cls_pred.size(2)
        default_boxes_xy = cxcy_to_xy(self.default_boxes)
        num_defaults = default_boxes_xy.size(0)
        loc_t = torch.Tensor([batch_size, num_defaults, 4], requires_grad=False) # Tensor to be filled w/ endcoded location targets.
        conf_t = torch.LongTensor([batch_size, num_defaults], requires_grad=False) # Tensor to be filled w/ matched indices for conf preds.
        
        # Match each default box with the ground truth box of the highest jaccard
        # overlap (IoU), encode the bounding boxes, then return the matched indices
        # corresponding to both confidence and location preds.
        for i in range(batch_size):
            n_objects = gt_boxes[i].size(0)
            overlaps = compute_iou(gt_boxes[i], default_boxes_xy) # iou size:(num_gt_boxes, num_default_boxes)
            overlaps_per_default_box, objects_per_default_box = overlaps.max(dim=0) 
            _, defaults_per_gt_boxes = overlaps.max(dim=1)
            objects_per_default_box[defaults_per_gt_boxes] = torch.LongTensor(range(n_objects)).to(self.device)
            overlaps_per_default_box[defaults_per_gt_boxes] = 1.
            label_each_default_box = gt_labels[i][objects_per_default_box]
            label_each_default_box[overlaps_per_default_box < self.threshold] = 0
            
            conf_t[i] = label_each_default_box
            loc_t[i] = encode_bboxes(xy_to_cxcy(gt_boxes[i][object_each_default_box]), self.default_boxes)
        
        loc_t = loc_t.to(self.device)
        conf_t = conf_t.to(self.device)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        #Localization loss
        #Localization loss is computed only over positive default boxes
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loc_loss = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        

