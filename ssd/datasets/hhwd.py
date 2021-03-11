"""
Hard Hat Workers Dataset:
https://public.roboflow.com/object-detection/hard-hat-workers/2
COCO annotation format
"""
import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np 
from PIL import Image
from pycocotools.coco import COCO
from ssd.datasets.augmentation import SSDAugmentation

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        super(COCOAnnotationTransform, self).__init__()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width  (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = obj['category_id']+1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")
        res = np.asarray(res)
        return res # [xmin, ymin, xmax, ymax] [label_idx]


class HHWDataset(Dataset):

    def __init__(self,
                 dataset_pth,
                 transform = SSDAugmentation(),
                 target_transform = COCOAnnotationTransform(),
                 mode = 'train'):
        super(HHWDataset, self).__init__()
        self.dataset_pth = dataset_pth
        self.mode = mode
        assert self.mode in ['train', 'test']
        self.root = os.path.join(self.dataset_pth, self.mode)
        print(os.path.join(self.dataset_pth, f'_{mode}_annotations.coco.json'))
        self.coco = COCO(os.path.join(self.dataset_pth, f'_{mode}_annotations.coco.json'))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(path), f'Image path does not exist: {path}'
        img = Image.open(path, mode="r")
        img = img.convert("RGB")
        height, width = img.size
        
        res = self.target_transform(target, width, height)
        boxes = torch.from_numpy(res[:, :4])
        labels = torch.from_numpy(res[:, 4])

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, boxes, labels)

        return img, boxes, labels
