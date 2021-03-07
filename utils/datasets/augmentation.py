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
    image: A PIL/Tensor image
    
    Out: New image (PIL/Tensor)
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

class RandomFlip(object):
    """
    Flip image horizontally.
    image: a PIL/Tesor image
    boxes: Bounding boxes, a tensor of dimensions (n_objects, 4)
    
    Out: flipped image (A PIL/Tensor image), new boxes
    """
    def __call__(self, image, boxes, labels=None):
        if random.random() > 0.5:
            return image, boxes, labels 
        width = image.width
        height = image.height
        image = TF.hflip(image)
        #flip boxes 
        new_boxes[:, 0] = width - boxes[:, 0]
        new_boxes[:, 2] = width - boxes[:, 2]
        new_boxes[:, 1] = height - boxes[:, 1] 
        new_boxes[:, 13 = height - boxes[:, 3] 
        new_boxes = new_boxes[:, [2, 3, 0, 1]]
        return image, new_boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size 

    def __call__(self, image, boxes, labels=None):
        width = image.width 
        height = image.height
        image = TF.resize(image, (self.size, self.size))
        old_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
        new_boxes = boxes / old_dims  # percent coordinates

        new_dims = torch.FloatTensor([self.size]*4).unsqueeze(0)
        new_boxes = new_boxes * new_dims
        return image, new_boxes, labels

class LightNoise(object):
    def __call__(self, image, boxes=None, labels=None):
        if random.random() > 0.5:
            return image, boxes, labels
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
                 (1, 2, 0), (2, 0, 1), (2, 1, 0))
        swap = perms[random.randint(0, len(perms) - 1)]
        image = TF.to_tensor(image)
        image = image[swap, :, :]
        image = TF.to_pil_image(image)
        return image, boxes, labels

class Expand(object):
    def __init__(self, max_scale = 4):
        self.max_scale = max_scale

    def __call__(self, image, boxes, labels=None):
        image = TF.to_tensor(image)
        height, width = image.size(0), image.size(1)
        image = TF.to_pil_image(image)
        return image, boxes, labels

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