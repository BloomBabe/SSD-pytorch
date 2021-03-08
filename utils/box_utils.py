import torch
import math


def cxcy_to_xy(bboxes):
    """
        Convert bboxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
    """
    return torch.cat([bboxes[..., :2] - (bboxes[..., 2:]/2),
                      bboxes[..., :2] + (bboxes[..., 2:]/2)], 1)

def xy_to_cxcy(bboxes):
    '''
        Convert bboxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
        bboxes: Bounding boxes, a tensor of dimensions (n_object, 4)
        
        Out: bboxes in center coordinate
    '''
    return torch.cat([(bboxes[:, 2:] + bboxes[:, :2]) / 2,
                      bboxes[:, 2:] - bboxes[:, :2]], 1)

def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (..., 2): left top corner.
        right_bottom (..., 2): right bottom corner.
    Returns:
        area (...): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def intersect(boxes0, boxes1):
    """
    Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)
        
        Out: Intersection each of boxes1 w.r.t. each of boxes2, 
             a tensor of dimensions (n1, n2)
    """
    n0 = boxes0.size(0)
    n1 = boxes1.size(0)
    max_xy = torch.min(boxes0[..., 2:].unsqueeze(1).expand(n0, n1, 2),
                       boxes1[..., 2:].unsqueeze(0).expand(n0, n1, 2))
    min_xy = torch.max(boxes0[..., :2].unsqueeze(1).expand(n0, n1, 2),
                       boxes1[..., :2].unsqueeze(0).expand(n0, n1, 2))
    return area_of(min_xy, max_xy)

def compute_iou(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (n1, 4): ground truth boxes.
        boxes1 (n2, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (n1, n2): IoU values.
    """
    overlap_area = intersect(boxes0, boxes1)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:]).unsqueeze(1).expand_as(overlap_area)
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:]).unsqueeze(0).expand_as(overlap_area)
    return overlap_area / (area0 + area1 - overlap_area + eps)

def encode_bboxes(bboxes,  default_boxes):
    """
        Encode bboxes correspoding default boxes (center form)
        Out: Encodeed bboxes to 4 offset, a tensor of dimensions (n_defaultboxes, 4)
    """
    return torch.cat([(bboxes[:, :2] - default_boxes[:, :2]) / (default_boxes[:, 2:] / 10),
                      torch.log(bboxes[:, 2:] / default_boxes[:, 2:]) *5],1)

def decode_bboxes(offsets, default_boxes):
    """
        Decode offsets
    """
    return torch.cat([offsets[:, :2] * default_boxes[:, 2:] / 10 + default_boxes[:, :2], 
                      torch.exp(offsets[:, 2:] / 5) * default_boxes[:, 2:]], 1)

def combine(batch):
    """
        Combine these tensors of different sizes in batch.
        batch: an iterable of N sets from __getitem__()
    """
    images = []
    boxes = []
    labels = []
    
    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        
    images = torch.stack(images, dim= 0)
    return images, boxes, labels




    
