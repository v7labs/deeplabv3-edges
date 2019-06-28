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

def load_model(n_classes=1, pretrained=False, aux_loss=False):

    # model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained, aux_loss=aux_loss)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)  
   
    # modify last model layers
    inplanes = 2048
    classifier =     
    model.classifier = DeepLabHead(inplanes, n_classes)

    if aux_loss:
        inplanes = 1024
        model.aux_classifier = FCNHead(inplanes, n_classes)
    else:
        model.aux_classifier = None

    np_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'DeepLabv3 -- num. of learnable parameters: {np_model}') 

    return model

def get_transform(train):

    transforms = []
    transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return transforms

if __name__== "__main__":
    
    torch.cuda.empty_cache()

    dbroot = '/datasets/OpenImages/processedv4'
    dataset_train = SegDataset(os.path.join(dbroot, 'test'), get_transform(train=True))
    dataset_val = SegDataset(os.path.join(dbroot, 'validation'), get_transform(train=False))

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, drop_last=True, batch_size=48, shuffle=True, num_workers=12)
#         collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, drop_last=True, batch_size=48, shuffle=False, num_workers=12)
#         collate_fn=utils.collate_fn)

    print(f"Train set size: {len(data_loader_train.dataset)}, n_batches: {len(data_loader_train)}")
    print(f"Validation set size: {len(data_loader_val.dataset)}, n_batches: {len(data_loader_val)}")

    # model
    model = load_model(pretrained=True, aux_loss=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.nn.DataParallel(model).to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=9,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        engine.evaluate(model, data_loader_val, device, epoch, print_freq=10)


    # for i in range(len(dataset)):|
    #     it = dataset.__getitem__(i)