import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform
        self.resize = transforms.Resize((128, 128))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.annotations.iloc[index, 1])
        image = io.imread(image_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 0]))

        if self.transform:
            image = self.transform(image)
            image = self.resize(image)
        return image, y_label

def get_dataloaders(root, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = ImageDataset(csv_file="train.csv", root=root, transform=transform)
    val_dataset = ImageDataset(csv_file="val.csv", root=root, transform=transform)
    test_dataset = ImageDataset(csv_file="test.csv", root=root, transform=transform)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True, pin_memory=True)

    return train_dl, val_dl, test_dl