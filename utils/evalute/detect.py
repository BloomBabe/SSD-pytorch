import torch
import numpy as np 
import torch.nn.functional as F
import torchvision.ops.nms as nms
from utils.box_utils import *

def detect(bboxes_pred, cls_pred, default_bboxes, min_score, iou_threshold, top_k, device):
    """
    bboxes_pred: Bounding boxes predictions (batch_size, num_defaults, 4)
    cls_pred: Class prediction logits (batch_size, num_defaults, num_classes)
    default_bboxes: Default bounding boxes (num_defaults, 4)
    min_score: Minimum confedence score threshold
    iou_threshold: Discards all overlapping boxes with IoU > iou_threshold
    top_k: Keep only top_k predictions
    """
    batch_size = bboxes_pred.size(0)
    num_defaults = default_bboxes.size(0)
    num_classes = cls_pred.size(2)

    cls_scores = F.softmax(cls_pred, dim=2)
    assert num_defaults == bboxes_pred.size(1) == cls_pred.size(1)

    all_image_boxes = []
    all_image_labels = []
    all_image_scores = []

    for i in range(batch_size):
        image_boxes = []
        image_labels = []
        image_scores = []
        decoded_bboxes = cxcy_to_xy(decode_bboxes(bboxes_pred[i], default_bboxes)) 
        max_scores, best_label = cls_pred[i].max(dim=1)
        for c in range(num_classes):
            conf_scores = cls_pred[i][:, c] 
            mask_min_conf = conf_scores > min_score
            conf_scores = conf_scores[mask_min_conf]
            if conf_scores.size(0) == 0:
                continue
            cls_bboxes = decoded_bboxes[mask_min_conf]
            nms_ids = nms(cls_bboxes, conf_scores, iou_threshold)

            image_boxes.append(cls_bboxes[nms_ids])
            image_labels.append(torch.LongTensor(nms_ids.size(0) * [c]).to(device))
            image_scores.append(class_scores[nms_ids])

        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))
        
        image_boxes = torch.cat(image_boxes, dim= 0)   #(n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)
        
        if n_objects > top_k:
            image_scores, sort_index = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_index][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_index][:top_k]  # (top_k)
            
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)
            
    return all_images_boxes, all_images_labels, all_images_scores        

