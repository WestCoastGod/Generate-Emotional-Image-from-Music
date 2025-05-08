import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torchvision
import torchvision.transforms as transforms


class EmotionDataset(Dataset):
    def __init__(self, img_dir, label_csv, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert the integer file name to a string and append the .jpg extension
        img_name = f"{int(self.labels.iloc[idx, 0])}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Get valence and arousal values, normalize to [0, 1]
        v = torch.tensor(self.labels.iloc[idx, 1] / 9.0, dtype=torch.float32)
        a = torch.tensor(self.labels.iloc[idx, 2] / 9.0, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, v, a
