import torch
import numpy as np 
import sys
from ssd.utils.box_utils import compute_iou


def compute_statiscs(bboxes_pred,
                     lables_pred, 
                     scores_pred,
                     gt_bboxes, 
                     gt_labels, 
                     iou_threshold=0.5,
                     device=None):
    """
    Compute True Positives, Predictive scores, Predicted labels per image for AP

    bboxes_pred: Bounding boxes predictions (batch_size, N_pred, 4)
    lables_pred: Label prediction (batch_size, N_pred)
    scores_pred:
    gt_bboxes: Ground Truth bounding boxes (batch_size, N_gt, 4)
    gt_labels: Ground Truth labels (batch_size, N_gt)
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
        true_postives = torch.zeros((bboxes_pred[sample_id].size(0))).to(device)
        detected_box = list()
        for pred_id, (bbox_pred, label_pred) in enumerate(zip(bboxes_pred[sample_id], lables_pred[sample_id])):
            if len(detected_box) == len(gt_labels[sample_id]):
                break
            if label_pred not in gt_labels[sample_id]:
                continue
            overlaps = compute_iou(bbox_pred.unsqueeze(0), gt_bboxes[sample_id])
            iou, max_idx = torch.max(overlaps.squeeze(0), dim=0)
            print("\nlabel_pred\n", label_pred)
            print("\ngt_labels\n", gt_labels[sample_id])
            print("\nbbox_pred\n", bbox_pred)
            print("\ngt_bboxes\n", gt_bboxes[sample_id])
            print("\niou\n", iou)
            print("\n\n")
            if sample_id == 3:
                sys.exit()
            if iou >= iou_threshold and max_idx not in detected_box:
                true_postives[pred_id] = 1
                detected_box += [max_idx]
        metrics_per_image.append([true_postives, scores_pred[sample_id], lables_pred[sample_id]])
    return metrics_per_image 


def compute_mAP(true_positives,
                conf_scores,
                label_pred,
                gt_labels,
                cat_dict,
                device=None):
    """
    Compute the average precision 
    true_positives: True Positives
    conf_scores: 
    label_pred:
    gt_labels:
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ap_per_class = dict()
    mean_AP = 0.
    for cat in cat_dict.keys():
        name = cat_dict[cat]
        ap_per_class[name] = 0.

    idx = torch.argsort(conf_scores, descending=True)
    true_positives, conf_scores, label_pred = true_positives[idx], conf_scores[idx], label_pred[idx]
    
    classes = torch.unique(gt_labels)
    for cls in classes:
        idx = label_pred==cls
        num_gt = (gt_labels == cls).sum().item()
        num_pred = idx.sum()
        
        cumul_true_positives = torch.cumsum(true_positives[idx], dim=0)  
        cumul_false_positives = torch.cumsum(1-true_positives[idx], dim=0)  

        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10) 

        cumul_recall = cumul_true_positives / (num_gt + 1e-10)

        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        mean_AP += precisions.mean().item()
        ap_per_class[cat_dict[str(cls.item())]] = precisions.mean()

    return ap_per_class, mean_AP/len(classes)
    
class Metrics(object):
    def __init__(self, device=None):
        super(Metrics, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.reset()
    
    def reset(self):
        self.mean_loss = 0.
        self.mean_conf_loss = 0.
        self.mean_loc_loss = 0.
        self.metrics_per_batch = list()
        self.targets = list()

    def update(self, loss, loc_loss, conf_loss, metrics):
        self.mean_loss += loss
        self.mean_conf_loss += conf_loss
        self.mean_loc_loss += loc_loss
        self.metrics_per_batch += metrics
    
    def mean_metrics(self, length):
        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*self.metrics_per_batch))]
        self.targets = torch.LongTensor(self.targets).to(self.device)
        return (self.mean_loss/length, 
                self.mean_conf_loss/length, 
                self.mean_loc_loss/length, 
                true_positives, 
                pred_scores, 
                pred_labels)

            
    
