"""
Dataloader for 
COCO annotation format data
"""
import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np 
from PIL import Image
from pycocotools.coco import COCO
from ssd.datasets.augmentation import SSDAugmentation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSES = ('background', 'biker', 'car', 'pedestrian', 'trafficLight', 'truck')


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, label_file):
        super(COCOAnnotationTransform, self).__init__()
        self.label_map = self._labelmap(label_file)

    def _labelmap(self, label_file):
        label_map = {}
        labels = open(label_file, 'r')
        for line in labels:
            ids = line.split(',')
            label_map[int(ids[0])] = int(ids[1])
        return label_map

    def __call__(self, target):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        labels = []
        bboxes = []
        for obj in target:
            if 'bbox' in obj:
                if obj['category_id'] not in self.label_map.keys():
                        continue
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']]
                bboxes += [bbox]      # [xmin, ymin, xmax, ymax]
                labels += [label_idx] # [label_idx]
            else:
                print("no bbox problem!")
        if len(bboxes) == 0:
            print(target)
            raise ValueError("Targets is empty")
        bboxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)
        return bboxes, labels   # [xmin, ymin, xmax, ymax] [label_idx]

# def label_map(annotation_file):
#     """
#     """
#     ann = json.load(open(annotation_file, 'r'))
#     cat_dict = dict()
#     cat_dict["0"] = "Background"
#     for cat in ann["categories"]:
#         cat_dict[str(cat["id"]+1)] = cat["name"]
#     return cat_dict

class COCODataset(Dataset):
    def __init__(self,
                 dataset_pth,
                 transform = SSDAugmentation(),
                 mode = 'train'):
        super(COCODataset, self).__init__()
        self.dataset_pth = dataset_pth
        self.mode = mode
        assert self.mode in ['train', 'test']
        self.root = os.path.join(self.dataset_pth, self.mode)
        print(os.path.join(self.dataset_pth, f'{mode}_annotations.coco.json'))
        self.coco = COCO(os.path.join(self.dataset_pth, f'{mode}_annotations.coco.json'))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = COCOAnnotationTransform(os.path.join(BASE_DIR, 'USDC_labels.txt'))

        self.cat_dict = dict()
        for i, cls in enumerate(CLASSES):
            self.cat_dict[str(i)] = cls

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(path), f'Image path does not exist: {path}'
        img = Image.open(path, mode="r")
        img = img.convert("RGB")
        
        boxes, labels = self.target_transform(target)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, boxes, labels)
        return img, boxes, labels
