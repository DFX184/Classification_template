import torch
import pandas as pd
import config
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split,TensorDataset
from rich import print
from PIL import Image
import os
from torchvision import transforms
from prefetch_generator import BackgroundGenerator
import torchvision
import numpy as np
import albumentations as A
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

def create_dataloader(root = "./data",train_transform=None,val_transform = None):
    train_set = torchvision.datasets.ImageFolder(os.path.join(root, "train"), transform = Transforms(train_transform))
    val_set = torchvision.datasets.ImageFolder(os.path.join(root, "test"), transform = Transforms(val_transform))
    train_loader = DataLoaderX(train_set,shuffle=True,
                                batch_size =config.parameter['batch_size'],
                                num_workers = config.parameter['num_workers'],
                                pin_memory=True)
    val_loader   = DataLoaderX(val_set,shuffle=False,
                                batch_size=config.parameter['batch_size'],
                                num_workers = config.parameter['num_workers'],
                                pin_memory=True)
    return train_loader,val_loader

if __name__ == "__main__":
    param = config.parameter
    print("parameters : ",param)
    

