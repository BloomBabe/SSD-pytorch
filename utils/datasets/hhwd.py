"""
Hard Hat Workers Dataset:
https://public.roboflow.com/object-detection/hard-hat-workers/2
"""
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image

class HHWDataset(Dataset):

    def __init__(self):
        super(HHWDataset, self).__init__()

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass