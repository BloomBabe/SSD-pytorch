"""
Split Udacity Self Driving Car Dataset (COCO format) on train/test
https://public.roboflow.com/object-detection/self-driving-car
"""
import os
import json
import argparse
import random
import shutil
from tqdm import tqdm
random.seed(42)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", default="/content/data1/export/", help="Dataset directory path")
ap.add_argument("--new_dataset_root", default="/content/data2/", help="New dataset directory path")
ap.add_argument("--ann_path", default="/content/data1/export/_annotations.coco.json", help="Annotation file path")

args = ap.parse_args()
data_root = args.dataset_root
new_data_root = args.new_dataset_root
annotation_file = args.ann_path

if __name__ == '__main__':
    print("Old dataset: ", data_root)
    filenames = os.listdir(data_root) 
    filenames = [filemame for filemame in filenames if filemame.endswith('.jpg')]
    random.shuffle(filenames)
    test_files = filenames[-1500:]
    train_files = filenames[:-1500]
    
    ann = json.load(open(annotation_file, 'r'))
    print("Read annotations: ", annotation_file)

    if not os.path.exists(new_data_root):
        os.mkdir(new_data_root)
    print(f"Create new dataset directory at {new_data_root}")
    for mode in ['train', 'test']:
        files = train_files if mode=='train' else test_files

        new_ann = dict()
        new_ann["info"] = ann["info"]
        new_ann["licenses"] = ann["licenses"]
        new_ann["categories"] = ann["categories"]
        new_ann["images"] = list()
        new_ann["annotations"] = list()

        dir_pth = os.path.join(new_data_root, mode)
        if not os.path.exists(dir_pth):
            os.mkdir(dir_pth)
        print(f"Create {mode} subdirectory at {dir_pth}")

        print(f"Iterating over all {mode} images...")
        for file_id, filename in tqdm(enumerate(files)):
            shutil.copy(os.path.join(data_root, filename), os.path.join(dir_pth, filename))
            old_id = None
            for img_dict in ann["images"]:
                if img_dict["file_name"] != filename:
                    continue
                old_id = img_dict["id"]
                new_img_dict = img_dict.copy()
                new_img_dict["id"] = file_id
                new_ann["images"].append(new_img_dict)
            assert old_id != None

            for ann_dict in ann["annotations"]:
                if ann_dict["image_id"] != old_id:
                    continue
                new_ann_dict = ann_dict.copy()
                new_ann_dict["image_id"] = file_id
                new_ann_dict["id"] = len(new_ann["annotations"])
                new_ann["annotations"].append(new_ann_dict)
            os.remove(os.path.join(data_root, filename))

        ann_pth = os.path.join(new_data_root,f'{mode}_annotations.coco.json')
        print(f"Save new {mode} annotations in {ann_pth}")
        with open(ann_pth, 'w+') as f:
            json.dump(new_ann, f,  indent=4)






