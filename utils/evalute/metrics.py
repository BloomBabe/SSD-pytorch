import torch


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

    
    
