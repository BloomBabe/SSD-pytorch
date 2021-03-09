import torch
import numpy as np 
from utils.box_utils import compute_iou


def compute_statiscs(bboxes_pred,
                     lables_pred, 
                     scores_pred,
                     gt_bboxes, 
                     gt_labels, 
                     num_classes, 
                     iou_threshold=0.5,
                     device=None):
    """
    Compute True Positives, Predictive scores, Predicted labels per image for AP

    bboxes_pred: Bounding boxes predictions (batch_size, N_pred, 4)
    lables_pred: Label prediction (batch_size, N_pred)
    scores_pred:
    gt_bboxes: Ground Truth bounding boxes (batch_size, N_gt, 4)
    gt_labels: Ground Truth labels (batch_size, N_gt)
    num_classes: Number of classes
    iou_threshold:
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(bboxes_pred)==len(lables_pred)==len(gt_bboxes)==len(gt_labels)
    batch_size = len(bboxes_pred)
    metrics_per_image = list()
    for sample_id in range(batch_size):
        if bboxes_pred[sample_id].size(0) == 0 and gt_bboxes[sample_id].size(0) == 0:
            continue
        true_postives = torch.zeros((bboxes_pred[sample_id].size(0)))
        detected_box = list()
        for pred_id, (bbox_pred, label_pred) in enumerate(zip(bboxes_pred[sample_id], lables_pred[sample_id])):
            if len(detected_box) == len(gt_labels[sample_id]):
                break
            if label_pred not in gt_labels[sample_id]:
                continue
            overlaps = compute_iou(bbox_pred.unsqueeze(0), gt_bboxes[sample_id])
            iou, max_idx = torch.max(overlaps.squeeze(0), dim=0)
            if iou >= iou_threshold and max_idx not in detected_box:
                true_postives[pred_id] = 1
                detected_box += [max_idx]
        metrics_per_image.append([true_postives, scores_pred[sample_id], lables_pred[sample_id]])
        
    return metrics_per_image



def compute_mAP(bboxes_pred,
                lables_pred, 
                gt_bboxes, 
                gt_labels, 
                num_classes, 
                device=None):
    """
    bboxes_pred: Bounding boxes predictions (batch_size, N_pred, 4)
    lables_pred: Label prediction (batch_size, N_pred)
    gt_bboxes: Ground Truth bounding boxes (batch_size, N_gt, 4)
    gt_labels: Ground Truth labels (batch_size, N_gt)
    num_classes: Number of classes 
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(bboxes_pred)==len(lables_pred)==len(gt_bboxes)==len(gt_labels)
    return


    
    
