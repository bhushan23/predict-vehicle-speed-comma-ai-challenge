import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from data_loading import *
from models import *
from utils import *

# initialization
train_data = './data/train/'
label_file = './data/train.txt'
device_type = 'cpu'

if torch.cuda.is_available():
    device_type = 'cuda'

# data processing
train_dataset, val_dataset = get_data_set_split(train_data, label_file, 10)

train_dl = DataLoader(train_dataset, batch_size = 1)
val_dl   = DataLoader(val_dataset, batch_size = 1)

# Debugging show few images
img, label = next(iter(train_dl))
img_w, img_h = img.shape[2], img.shape[3]
print(img.shape)
# save_image(img)

# img, label = next(iter(val_dl))
# print(img.shape)
# save_image(img)

# training
model = DeepVO(img_w, img_h) #.type(device_type)
# train(model, train_dl, val_dl)