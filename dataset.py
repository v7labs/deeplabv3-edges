import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision


class SegDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        # cut_dataset_at=10000

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))#[:cut_dataset_at]
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))#[:cut_dataset_at]
        self.polys = list(sorted(os.listdir(os.path.join(root, "polygons"))))#[:cut_dataset_at]

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
        
        model_input_size = (300, 300)

        img = cv2.resize(img, model_input_size)
        mask = cv2.resize(mask, model_input_size)

        img = torchvision.transforms.ToTensor()(img)
        mask = torch.tensor(np.expand_dims(mask, axis=0), dtype=torch.float)

        # if self.transforms is not None:
        #     for transform in self.transforms:
        #         print(transform)
        #         img = transform(img)
        #         mask = transform(target)


        return img, mask

    def __len__(self):
        return len(self.imgs)