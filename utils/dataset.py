import torch
import torchvision
from torchvision import transforms
import os
import glob
from torch.utils.data import Dataset, DataLoader 
import cv2

class Custom_dataset(Dataset): 
  def __init__(self, root_path, mode, transform=None):
    self.all_data = sorted(glob.glob(os.path.join(root_path, mode, '*', '*'))) 
                                                                              
    self.transform = transform

  def __getitem__(self, index):
    if torch.is_tensor(index): 
      index = index.tolist()

    data_path = self.all_data[index]
    image = cv2.imread(data_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    if self.transform is not None:
     augmentation = self.transform(image =  image)
     image = augmentation['image']

    if 'A' in data_path.split('/')[-2]:
      label = 0
    elif 'BE' in data_path.split('/')[-2]:
      label = 1
    else:
      label = 2

    return image, label

  def __len__(self):
    length = len(self.all_data) 
    return length