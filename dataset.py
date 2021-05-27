import os

import pandas as pd
import torch
import rasterio
from torchvision.transforms import ToTensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, images_folder, masks_folder):
        self.samples = samples
        self.images_folder = images_folder
        self.masks_folder = masks_folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = rasterio.open(os.path.join(self.images_folder, f"/{self.samples[index]}.tif"))
        image = ToTensor()(image)

        mask = rasterio.open(os.path.join(self.masks_folder, f"/{self.samples[index]}.tif"))
        mask = ToTensor()(mask)

        return index, (image, mask)
