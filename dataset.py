import os

import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image


def convert_data(filenames, path_to_files):
    image = []
    labels = []

    for file in filenames:
        for c in os.listdir(os.path.join(path_to_files, file)):
            image.append(c)
            labels.append(file)
    data = {'Images': image, 'labels': labels}
    data = pd.DataFrame(data)
    return data


def encode_labels(data):
    lb = LabelEncoder()
    data['encoded_labels'] = lb.fit_transform(data['labels'])
    return data


def split_dataset(data, test_split, random_seed, shuffle_dataset):
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler, test_sampler


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class TechDataset(Dataset):
    def __init__(self, img_data, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.img_data.loc[index, 'labels'],
                                self.img_data.loc[index, 'Images'])
        image = Image.open(img_name)
        image = image.resize((300, 300))
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_data(dataset, batch_size, train_sampler, test_sampler):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler)
    return train_loader, test_loader
