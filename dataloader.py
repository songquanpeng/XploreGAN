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

    def __init__(self, cluster_npz_path, dataset_path, transform, mode, selected_clusters=None, seed=None):
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
        self.image_paths = [os.path.join(dataset_path, i) for i in self.image_paths]
        self.seed = 2333 if seed is None else seed
        self.selected_clusters = [] if selected_clusters is None else [int(x) for x in selected_clusters]
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
            print(f"Image number for training: {self.num_images}")
        else:
            self.num_images = len(self.test_dataset)
            print(f"Image number for testing: {self.num_images}")

    def preprocess(self):
        if len(self.selected_clusters) == 0:
            images = [(self.image_paths[i], label, self.centers[label], self.stds[label]) for i, label in
                      enumerate(self.labels)]
            random.seed(self.seed)
            random.shuffle(images)
            self.test_dataset = images[:2000]
            self.train_dataset = images[2000:]
        else:
            train_images = []
            test_images = []
            for i, label in enumerate(self.labels):
                image = (self.image_paths[i], label, self.centers[label], self.stds[label])
                if label in self.selected_clusters:
                    train_images.append(image)
                else:
                    test_images.append(image)
            random.seed(self.seed)
            random.shuffle(train_images)
            random.shuffle(test_images)
            self.test_dataset = train_images
            self.train_dataset = test_images
        print('Dataset processed.')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        path, label, center, std = dataset[index]
        image = Image.open(path)
        # TODO: is it okay to directly return variables like that?
        return self.transform(image), label, center, std

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(cluster_npz_path, dataset_path, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, selected_clusters=None, seed=None):
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
        dataset = CelebA(cluster_npz_path, dataset_path, transform, mode, selected_clusters, seed)
    # elif dataset == 'RaFD':
    #     dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader
