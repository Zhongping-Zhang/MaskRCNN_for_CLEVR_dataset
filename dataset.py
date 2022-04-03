# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:31:28 2020

@author: zhongping zhang
"""
import os
import glob
import json
import numpy as np
import torch
from PIL import Image

import torchvision.transforms as transforms
from matplotlib.pyplot import imshow,figure
import detection.utils as utils
import detection.transforms as T

import skimage.io
from skimage import color
from skimage.filters import threshold_mean

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224), transforms.CenterCrop(224),
                transforms.ToTensor(), normalize
            ])

class CLEVRsegmentDataset2(object):
    def __init__(self, root, transforms=None,category=False):
        self.root = root
        self.transforms = transforms
        self.category=category
        
        with open(os.path.join(root,"CLEVR_scenes.json"),'r') as f:
            self.scenes = json.load(f)['scenes']
        with open(os.path.join(root,"vocab.json"),'r') as f:
            self.vocab = json.load(f)
            self.obj2idx = self.vocab['object_name_to_idx']
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        
        img_path = os.path.join(self.root, "images", scene['image_filename'])
        img = Image.open(img_path).convert("RGB")

        mask_dir = os.path.join(img_path[:-4],'mask')
        mask = []
        for f in sorted(next(os.walk(mask_dir))[2]):
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f))
                m = color.rgb2gray(m)
                thresh = threshold_mean(m)
                m = m > thresh
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        masks = np.transpose(mask,(2,0,1)) # (num_obj,height,width)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        objects = scene['objects']
        objectnames = []
        for obj in objects:
            objectnames.append(obj['color']+" "+obj['shape'])
        if self.category==True:
            labels = [self.obj2idx[objname] for objname in objectnames]
            labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        else:
            labels = torch.ones((len(objectnames),), dtype=torch.int64)
        
        num_objs = len(objectnames)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id # idx of __getitem__ 
        target["area"] = area # area of box
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
            
        return img, target

    def __len__(self):
        return len(self.scenes)
    
    
class CLEVRsegmentDataset(object):
    def __init__(self, root, transforms=None,category=False):
        self.root = root
        self.transforms = transforms
        self.category=category
        
        # with open(os.path.join(root,"CLEVR_scenes.json"),'r') as f:
        #     self.scenes = json.load(f)['scenes']
        
        self.scenes = glob.glob(root+"/scenes/*")
        
        with open(os.path.join(root,"vocab.json"),'r') as f:
            self.vocab = json.load(f)
            self.obj2idx = self.vocab['object_name_to_idx']
    
    def __getitem__(self, idx):
        with open(self.scenes[idx],'r') as f:
            scene = json.load(f)
        # scene = self.scenes[idx]
        # print(scene["image_filename"],scene["objects"])
        img_path = os.path.join(self.root, "images", scene['image_filename'])
        img = Image.open(img_path).convert("RGB")

        mask_dir = os.path.join(img_path[:-4],'mask')
        mask = []
        for f in sorted(next(os.walk(mask_dir))[2]):
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f))
                m = color.rgb2gray(m)
                thresh = threshold_mean(m)
                m = m > thresh
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        masks = np.transpose(mask,(2,0,1)) # (num_obj,height,width)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        objects = scene['objects']
        objectnames = []
        for obj in objects:
            objectnames.append(obj['color']+" "+obj['shape'])
        if self.category==True:
            labels = [self.obj2idx[objname] for objname in objectnames]
            labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        else:
            labels = torch.ones((len(objectnames),), dtype=torch.int64)
        
        num_objs = len(objectnames)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id # idx of __getitem__ 
        target["area"] = area # area of box
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
            
        return img, target

    def __len__(self):
        return len(self.scenes)
    
    
if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_test = CLEVRsegmentDataset("output", get_transform(train=False))
    
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    images,targets = next(iter(data_loader_test))
    imshow(np.transpose(images[0].numpy(),(1,2,0)))
    
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    
    """
    images: list images[i]=>(3,height,width) tensor
    targets: list 
        targets[i] =>
            target["boxes"] = boxes (num_obj, 4)
            target["labels"] = labels (num_obj)
            target["masks"] = masks (num_obj, height,width)
            target["image_id"] = image_id (1)
            target["area"] = area (num_obj)
            target["iscrowd"] = iscrowd (num_obj)
    
    
    
    """
    
    
    
    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    
