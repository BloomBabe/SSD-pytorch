"""
Inference model with single image
"""
import torch
import os
import numpy as np 
import argparse
import random
from PIL import Image, ImageDraw, ImageFont
from ssd.datasets.augmentation import SSDDetectAug
from ssd.evalute.detect import detect
from ssd.datasets.dataloader import CLASSES
from matplotlib.pyplot import imshow

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--trained_model", default = "/content/drive/MyDrive/SSD/15.03/checkpoints/ssd300_epoch33_lr0.001_2021-03-16 17:56:33.pth.tar", type= str,
                    help = "Trained state_dict file path to open")
parser.add_argument("--img_pth", type = str, help = "Path to input image")
parser.add_argument("--output_pth", type = str, help = "Path to save output image")
parser.add_argument("--min_score", default = 0.1, type = float, help = "Min score for detect")
parser.add_argument("--iou_threshold", default = 0.5, type = float, help = "Max overlap for NMS")
parser.add_argument("--top_k", default = 200, type = int, help = "Top k for NMS")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.trained_model, map_location=device)
model = model["model"]
model = model.to(device)
model.eval()

img_pth = args.img_pth
output_pth = args.output_pth
min_score = args.min_score
iou_threshold = args.iou_threshold
top_k = args.top_k

def drawPred(image, bbox_pred, labels_pred, conf_scores):
    distinct_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in CLASSES]
    label_color_map  = {k: distinct_colors[i] for i, k in enumerate(CLASSES)}
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', 15) 
    for i in range(bbox_pred.size(0)):
        bbox = bbox_pred[i].tolist()
        label = labels_pred[i]
        conf = conf_scores[i]
        draw.rectangle(xy=bbox, outline=label_color_map[label])
        draw.rectangle(xy=[l + 1. for l in bbox], outline=label_color_map[label])
        
        conf = f" {conf:.2f}"
        text_size = font.getsize(label.upper()+conf)
        text_location = [bbox[0] + 2., bbox[1] - text_size[1]]
        textbox_location = [bbox[0], bbox[1] - text_size[1], bbox[0] + text_size[0] + 4., bbox[1]]

        draw.rectangle(xy=textbox_location, fill=label_color_map[label])
        draw.text(xy=text_location, text=label.upper()+conf, fill='white', font=font)
    
    return image


if __name__ == '__main__':
    # Read image and store its dimensions
    image = Image.open(img_pth, mode='r')
    image = image.convert('RGB')
    width = image.width
    height = image.height
    orig_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0).to(device)
    # Transformed image by defaults val augmentations
    transformed_img = image.copy()
    transformed_img, _, _ = SSDDetectAug()(transformed_img)
    transformed_img = transformed_img.unsqueeze(0)
    # Make predictions and decode them
    cls_pred, locs_pred = model(transformed_img.to(device))
    locs_pred, labels_pred, conf_scores = detect(locs_pred, cls_pred, 
                                                model.default_bboxes, 
                                                min_score=0.4,
                                                iou_threshold=0.5,
                                                top_k=top_k,
                                                device=device)

    locs_pred = locs_pred[0]*orig_dims
    labels_pred = [CLASSES[i] for i in labels_pred[0].tolist()]
    if len(labels_pred) == 1 and labels_pred[0] == 'background':
        raise ValueError ('No predictions')
    conf_scores = conf_scores[0]
    image = drawPred(image, locs_pred, labels_pred, conf_scores)
    image.save(os.path.join(output_pth,'pred_img.jpg'))
    imshow(np.asarray(image))
    




