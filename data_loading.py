import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import glob
import cv2
from random import randint
import os
from skimage import io
from PIL import Image

HEIGHT = 250
WIDTH = 450


def get_data_set_split(dir, label_file, validation_split):
    images = []
    labels = []

    for img in sorted(glob.glob(dir + '/*.jpg'), key=os.path.getmtime):
        images.append(img)

    with open(label_file) as f:
        label_file = f.read().splitlines()
        for each in label_file:
            labels.append(float(each))
    
    validation_count = int (validation_split * len(images) / 100)
    split_index = randint(0, len(images) - validation_count - 1)
    
    # Build train and val set
    train_images = images[:split_index] +  images[split_index+validation_count:]
    val_images = images[split_index : split_index + validation_count]
    
    train_labels = labels[:split_index] + labels[split_index+validation_count:]
    val_labels = labels[split_index : split_index + validation_count]
    
    # Build custom datasets
    transform = transforms.Compose([
                transforms.CenterCrop((HEIGHT, WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
    train_dataset = DrivingDataset(train_images, train_labels, transform = transform)
    val_dataset = DrivingDataset(val_images, val_labels, transform = transform)

    return train_dataset, val_dataset
    

class DrivingDataset(Dataset):
    def __init__(self, images, labels, transform, num_of_images = 2, is_test = False):
        self.images = images
        self.labels = labels
        self.is_test = is_test
        self.transform = transform
        self.num_of_images = num_of_images
    
    def __getitem__(self, index):
        # img = io.imread(self.images[index])
        # img = cv2.imread(self.images[index], )
        img = None
        if index - self.num_of_images + 1 < 0:
            img = torch.zeros(abs(index - self.num_of_images + 1) * 3, HEIGHT, WIDTH)

        for i in range(max(0, index - self.num_of_images + 1), index + 1):
            temp = self.transform(Image.open(self.images[i]))
            if img is None:
                img = temp
            else:
                img  = torch.cat((img, temp))

        if self.is_test:
           return img, None
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images) # - self.num_of_images
