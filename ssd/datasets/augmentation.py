import torch
from torchvision import transforms
import numpy as np
import types
import random
import torchvision.transforms.functional as TF
import PIL
from ssd.utils.box_utils import compute_iou

random.seed(42)

class NormalizeBox(object):
    def __call__(self, image, boxes, labels=None):
        width = image.width 
        height = image.height 
        boxes[:, 0] = boxes[:, 0]/width
        boxes[:, 1] = boxes[:, 1]/height
        boxes[:, 2] = boxes[:, 2]/width
        boxes[:, 3] = boxes[:, 3]/height
        return image, boxes, labels

class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None):
        return TF.to_tensor(image), boxes, labels

class Normalize(object):
    def __init__(self, 
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]):
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
        if type(image) != PIL.Image.Image:
            image = TF.to_pil_image(image)      
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
        if type(image) != PIL.Image.Image:
            image = TF.to_pil_image(image)
        #flip boxes 
        new_boxes = boxes.detach().clone()
        new_boxes[:, 0] = width - boxes[:, 0]
        new_boxes[:, 2] = width - boxes[:, 2]
        new_boxes[:, 1] = height - boxes[:, 1] 
        new_boxes[:, 3] = height - boxes[:, 3] 
        new_boxes = new_boxes[:, [2, 3, 0, 1]]
          
        return image, new_boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size 

    def __call__(self, image, boxes=None, labels=None):
        width = image.width 
        height = image.height
        image = TF.resize(image, (self.size, self.size))
        if type(image) != PIL.Image.Image:
            image = TF.to_pil_image(image)
        if boxes is None:
            return image, new_boxes, labels
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
    def __init__(self, max_scale = 4, mean=[0.485, 0.456, 0.406]):
        self.max_scale = max_scale
        self.mean = mean

    def __call__(self, image, boxes, labels=None):
        if random.random() < 0.5:
            return image, boxes, labels
        width = image.width 
        height = image.height
        image = TF.to_tensor(image)
        scale = random.uniform(1, self.max_scale)
        new_h = int(scale*height)
        new_w = int(scale*width)

        filler = torch.FloatTensor(self.mean) #(3)
        new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

        # Place the original image at random coordinates 
        #in this new image (origin at top-left of image)
        left = random.randint(0, new_w - width)
        right = left + width
        top = random.randint(0, new_h - height)
        bottom = top + height

        new_image[:, top:bottom, left:right] = image
        new_image = TF.to_pil_image(new_image)
        #Adjust bounding box
        new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
            
        return new_image, new_boxes, labels

class RandomCrop(object):
    """Crop
    https://github.com/amdegroot/ssd.pytorch/blob/5b0b77faa955c1917b0c710d770739ba8fbff9b7/utils/augmentations.py#L67
    Arguments:
        img (PIL Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
        img (Image): the cropped image
        boxes (Tensor): the adjusted bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
    def __call__(self, image, boxes, labels):
        width = image.width
        height = image.height
        image = TF.to_tensor(image)
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                image = TF.to_pil_image(image)
                return image, boxes, labels
            
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            # max trails (50)
            for _ in range(50):
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # convert to integer rect x1,y1,x2,y2
                crop = torch.FloatTensor([int(left), int(top), int(left+w), int(top+h)])
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = compute_iou(boxes, crop)
                overlap = overlap.squeeze(0)
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min().item() < min_iou and max_iou < overlap.max().item():
                    continue
                # cut the crop from the image
                new_image = image[:, int(top):int(top+h), int(left):int(left+w)]
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (crop[0] < centers[:, 0]) * (crop[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (crop[2] > centers[:, 0]) * (crop[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                # take only matching gt boxes
                current_boxes = boxes[mask, :].detach().clone()
                # take only matching gt labels
                current_labels = labels[mask]
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = torch.max(current_boxes[:, :2],
                                                  crop[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= crop[:2]
                current_boxes[:, 2:] = torch.min(current_boxes[:, 2:],
                                                  crop[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= crop[:2]
                
                new_image = TF.to_pil_image(new_image)
                return new_image, current_boxes, current_labels

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels
        
class SSDAugmentation(object):
    def __init__(self, size=300, mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            Distort(),
            LightNoise(),
            Expand(max_scale=4, mean=mean),
            RandomCrop(),
            RandomFlip(),
            Resize(size=size),
            NormalizeBox(),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class SSDDetectAug(object):
    def __init__(self, size=300, mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            Resize(size=size),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, boxes=None, labels=None):
        return self.augment(img, boxes, labels)
