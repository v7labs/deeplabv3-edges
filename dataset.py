import os
import cv2
import numpy as np
import torch
from PIL import Image

class SegDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        self.polys = list(sorted(os.listdir(os.path.join(root, "polygons"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        poly_path = os.path.join(self.root, "polygons", self.polys[idx])
        
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        
        polys = np.load(poly_path)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        mask =  np.zeros((img.shape[0], img.shape[1]))
        cv2.drawContours(mask, [polys], -1, 1, 5)
        mask = np.expand_dims(mask, axis=0)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        print(img.shape, mask.shape)
            
        return img, mask

    def __len__(self):
        return len(self.imgs)