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
        labels = []
        bboxes = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = obj['category_id']+1
                bboxes += [bbox]      # [xmin, ymin, xmax, ymax]
                labels += [label_idx] # [label_idx]
            else:
                print("no bbox problem!")
        bboxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)
        return bboxes, labels # [xmin, ymin, xmax, ymax] [label_idx]

def label_map(annotation_file):
    """
    Return categories dict:
    {"0": "Background",
     "1": "Workers",
     "2": "head"
     ...}
    """
    ann = json.load(open(annotation_file, 'r'))
    cat_dict = dict()
    cat_dict["0"] = "Background"
    for cat in ann["categories"]:
        cat_dict[str(cat["id"]+1)] = cat["name"]
    return cat_dict

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
        
        self.cat_dict = label_map(os.path.join(self.dataset_pth, f'_{mode}_annotations.coco.json'))
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
        
        boxes, labels = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, boxes, labels)

        return img, boxes, labels
