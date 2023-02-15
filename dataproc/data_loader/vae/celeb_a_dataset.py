from typing import *
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

RANDOM_SEED = 1

class CelebADataset(Dataset):
    def __init__(self,
                 list_of_data_paths: str,
                 patch_size: Union[int, Sequence[int]] = (256, 256),
                ):

        self.customed_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        self.dataset = list_of_data_paths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        if not os.path.exists(img_path):
            raise Exception
        img = Image.open(img_path)
        data_tensor = self.customed_transforms(img)
        return data_tensor
