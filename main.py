import os
import torch
import torchvision
from pathlib import Path
from torchvision.models.segmentation import deeplabv3_resnet101

import engine
import transforms as T
import utils

from dataset import SegDataset
from open_images import OpenImagesDataset

from models.deeplabv3 import DeepLabHead
from models.fcn import FCNHead

def load_model(pretrained=False):
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)        
    np_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'DeppLabv3 -- num. of learnable parameters: {np_model}') 

    return model

def get_transform(train):

    transforms = []
    transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__== "__main__":

    dbroot = '/datasets/OpenImages/processedv4'
    dataset_train = SegDataset(os.path.join(dbroot, 'test'), get_transform(train=True))
    dataset_val = SegDataset(os.path.join(dbroot, 'validation'), get_transform(train=False))

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,)
        # collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,)
        # collate_fn=utils.collate_fn)

    # model
    model = load_model(pretrained=True)
    num_classes = 1
    aux=True
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    inplanes = 2048
    classifier = DeepLabHead(inplanes, num_classes)
    
    model.classifier = classifier
    model.aux_classifier = aux_classifier

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=9,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 30

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)


    # for i in range(len(dataset)):
    #     it = dataset.__getitem__(i)