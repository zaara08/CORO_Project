from __future__ import print_function, division
import os
from os.path import join
import numpy as np
import torch
import pandas as pd
import skimage
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import warnings
warnings.filterwarnings("ignore")



def show_position(image, label):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(label[0], label[1], s=10, marker='.', c='r')
    plt.pause(0.001)  


class EndEffectorPositionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, load_data_and_labels_from_same_folder=False, use_cuda=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.load_data_and_labels_from_same_folder = load_data_and_labels_from_same_folder

    def __len__(self):
        if not self.load_data_and_labels_from_same_folder:
            return len(os.listdir(join(self.root_dir, 'images')))
        else:
            return len([f for f in os.listdir(self.root_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    def __getitem__(self, idx):
        if not self.load_data_and_labels_from_same_folder:
            img_name = join(self.root_dir,
                                    'images', '{0:06d}.png'.format(idx))
            image = io.imread(img_name)
            if image.shape[-1] == 4:
                image = image[:, :, :-1]
           
            label = np.load(join(self.root_dir, 'labels', '{0:06d}.npy'.format(idx)))[:2] 
            label = np.round(label).astype(np.int32)
        else:
            img_name = join(self.root_dir,
                                '{0:06d}.png'.format(idx))
            image = io.imread(img_name)
            if image.shape[-1] == 4:
                image = image[:, :, :-1]
           
            label = np.load(join(self.root_dir, 
                                '{0:06d}.npy'.format(idx)))[:2] 
            label = np.round(label).astype(np.int32)            
        label = label[0] 
        buff = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)
        buff[label[1], label[0]] = 1
        label = buff
        image = skimage.img_as_float32(image)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample




