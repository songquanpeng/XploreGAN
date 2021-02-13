from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, cluster_npz_path, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        clusters = np.load(cluster_npz_path)
        self.centers = clusters["centers"]
        self.labels = clusters["labels"]
        self.stds = clusters["stds"]
        self.image_paths = clusters["image_paths"]
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        images = [(self.image_paths[i], label) for i, label in enumerate(self.labels)]
        random.seed(1234)
        random.shuffle(images)
        self.test_dataset = images[:2000]
        self.train_dataset = images[2000:]
        print('Dataset processed.')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        path, label = dataset[index]
        label = [label]
        image = Image.open(path)
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(cluster_npz_path, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(cluster_npz_path, transform, mode)
    # elif dataset == 'RaFD':
    #     dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader
