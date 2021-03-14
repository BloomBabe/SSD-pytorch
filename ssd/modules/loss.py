import torch.nn as nn 
import torch.nn.functional as F
from ssd.utils.box_utils import *
import sys

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
        loc_t = torch.zeros([batch_size, num_defaults, 4], requires_grad=False) # Tensor to be filled w/ endcoded location targets.
        conf_t = torch.zeros([batch_size, num_defaults], dtype=torch.long, requires_grad=False) # Tensor to be filled w/ matched indices for conf preds.
        
        # Match each default box with the ground truth box of the highest jaccard
        # overlap (IoU), encode the bounding boxes, then return the matched indices
        # corresponding to both confidence and location preds.
        for i in range(batch_size):
            n_objects = gt_boxes[i].size(0)
            overlaps = compute_iou(gt_boxes[i], default_boxes_xy) # iou size:(num_gt_boxes, num_default_boxes)
            
            # Find the gt_bboxes idxes corresponding to default bboxes by IoU
            best_truth_overlap, best_truth_idx  = overlaps.max(dim=0) 

            # For gt_bboxes, we find the indices of default bboxes corresponding
            # to them according to IoU (best_default_idx), and for these default
            # bboxes we replace the previous indices (in best_truth_idx) with 
            # these indices of the best boxes 
            best_default_overlap, best_default_idx = overlaps.max(dim=1)
            best_truth_idx[best_default_idx] = torch.LongTensor(range(n_objects)).to(self.device)

            # For the found pairs of gt_bboxes and default ones, replace them IoU by 1.
            best_truth_overlap[best_default_idx] = 1.
            
            label_each_default_box = gt_labels[i][best_truth_idx]
            label_each_default_box[best_truth_overlap < self.threshold] = 0
            conf_t[i] = label_each_default_box
            loc_t[i] = encode_bboxes(xy_to_cxcy(gt_boxes[i][best_truth_idx]), self.default_boxes)
        
        loc_t = loc_t.to(self.device)
        conf_t = conf_t.to(self.device)

        pos_labels = conf_t > 0 # (batch_size, num_default_boxes)
        # Localization loss
        # Localization loss is computed only over positive default boxes
        smooth_L1_loss = nn.SmoothL1Loss()
        loc_loss = smooth_L1_loss(loc_pred[pos_labels], loc_t[pos_labels])
        
        # Number of positive ad hard-negative default boxes per image
        n_positive = pos_labels.sum(dim=1)
        n_hard_negatives = self.neg_pos * n_positive
        
        # Find the loss for all priors
        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        confidence_loss_all = cross_entropy_loss(cls_pred.view(-1, num_classes), conf_t.view(-1)) # (batch_size*num_default_boxes)
        confidence_loss_all = confidence_loss_all.view(batch_size, num_defaults) # (batch_size, num_default_boxes)
        
        confidence_pos_loss = confidence_loss_all[pos_labels]
        
        # Find which priors are hard-negative
        confidence_neg_loss = confidence_loss_all.clone() # (batch_size, num_default_boxes)
        confidence_neg_loss[pos_labels] = 0.
        confidence_neg_loss, _ = confidence_neg_loss.sort(dim=1, descending=True)
        
        hardness_ranks = torch.LongTensor(range(num_defaults)).unsqueeze(0).expand_as(confidence_neg_loss).to(self.device)  # (batch_size, num_default_boxes)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1) # (batch_size, num_default_boxes)
        confidence_hard_neg_loss = confidence_neg_loss[hard_negatives] 
        confidence_loss = (confidence_hard_neg_loss.sum() + confidence_pos_loss.sum()) / n_positive.sum().float()
        
        return self.alpha * loc_loss + confidence_loss, loc_loss, confidence_loss

