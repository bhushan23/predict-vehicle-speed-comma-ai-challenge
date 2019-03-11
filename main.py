import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loading import *
import matplotlib.pyplot as plt
import torchvision

from utils import *
# initialization
train_data = './data/train/'
label_file = './data/train.txt'

# data processing
train_dataset, val_dataset = get_data_set_split(train_data, label_file, 10)

train_dl = DataLoader(train_dataset, batch_size = 1)
val_dl = DataLoader(val_dataset, batch_size = 1)

img, label = next(iter(train_dl))
print(img.shape)
save_image(img)

img, label = next(iter(val_dl))
print(img.shape)
save_image(img)

# training