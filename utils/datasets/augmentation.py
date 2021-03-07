import torch
from torchvision import transforms
import numpy as np
import types
import random
import torchvision.transforms.functional as TF
import PIL

random.seed(42)

class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None):
        return TF.to_tensor(image), boxes, labels

class Normalize(object):
    def __init__(self, 
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225])
        self.mean = mean 
        self.std = std 

    def __call__(self, image, boxes=None, labels=None):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, boxes, labels

class Distort(object):
    """
    Distort brightness, contrast, saturation
    image: A PIL image
    
    Out: New image (PIL)
    """
    def __call__(self, image, boxes=None, labels=None):
        distortions = [TF.adjust_brightness,
                       TF.adjust_contrast,
                       TF.adjust_saturation]
    
        random.shuffle(distortions)
        for function in distortions:
            if random.random() < 0.5:
                adjust_factor = random.uniform(0.5, 1.5)
                image = function(image, adjust_factor)       
        return image, boxes, labels

class 

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)