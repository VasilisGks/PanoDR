import os
import cv2
import numpy as np
import random
import torch
import time
import glob
from PIL import Image
from torch.utils.data.dataset import Dataset
from vcl3datlantis.dataloaders.layout2sem import one_hot
from typing import Set,Tuple, Dict
from torch.utils.data import DataLoader
import torchvision.transforms as T
from vcl3datlantis.dataloaders.misc.utils import showImages, colorizeLabels

class M3D(Dataset):
    def __init__(self, 
        root_path: str,
        num_classes: int,
        size: Tuple
        )->None:
        self.root_path = root_path
        self.img_paths = glob.glob(f"{self.root_path}/*emission_center*")
        self.layout_paths = glob.glob(f"{self.root_path}/*layout*.exr*")
        self.image_size = size
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        img = Image.open(img_name)
        img  = self.transform(img)
        
        layout = cv2.imread(self.layout_paths[idx], cv2.IMREAD_UNCHANGED)[:,:,0]
        layout  = self.transform(layout)
        one_hot_t = torch.zeros(1, self.num_classes, self.image_size[1], self.image_size[0])
        one_hot_t.scatter_(1, layout.unsqueeze(0).long(), 1).squeeze_(0)

        data = {"img": img, "layout": layout, "layout_one_hot": one_hot_t}
        return data


if __name__ == "__main__":

    image_size = [512,256]
    num_classes = 4
    dataset = M3D("D:/Data/subset_m3d/", num_classes, image_size)
    dataloader = DataLoader(dataset, batch_size = 2, shuffle=True)
    counter = 0
    timer = 0
    torch.manual_seed(0)
    for batch in dataloader:
        img = batch["img"]
        layout = batch["layout"]
        layout_one_hot = batch["layout_one_hot"]
        b,c,h,w = layout.size()
        # showImages(
        #     img = img[0].permute(1,2,0).numpy()
        #     #masked = masked[0].permute(1,2,0).numpy(),
        #     #gt = img_gt[0].permute(1,2,0).numpy(),
        #     #mask = mask[0].permute(1,2,0).numpy(),
        # )
        