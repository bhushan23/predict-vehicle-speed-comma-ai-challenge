import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from data_loading import *
from models import *
from utils import *
from train import *

# initialization
train_data = './data/train/'
label_file = './data/train.txt'
test_data  = './data/test/'

device_type = 'cpu'
if torch.cuda.is_available():
    device_type = 'cuda'

# data processing
train_dataset, val_dataset = get_data_set_split(train_data, label_file, 10)

train_dl = DataLoader(train_dataset, batch_size = 2)
val_dl   = DataLoader(val_dataset, batch_size = 2)
print('Train data: ', len(train_dl), ' Val data: ', len(val_dl))

# Debugging show few images
img, label = next(iter(train_dl))
img_w, img_h = img.shape[2], img.shape[3]
print(img.shape)
# save_image(img)

# img, label = next(iter(val_dl))
# print(img.shape)
# save_image(img)

# training
model = DeepVO_small(img_w, img_h).to(device_type)
train(model, train_dl, val_dl, test_data = test_data, num_epochs = 20)

test_dataset = get_test_dataset(test_data)
test_dl = DataLoader(test_dataset, batch_size = 5)
test(model, test_dl, write_to_file='test_output.csv')
