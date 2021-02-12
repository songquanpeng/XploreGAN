import os
import argparse
import numpy as np
import torch
from torch.backends import cudnn
from pretrained import bagnet
from torchvision import transforms as T
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import math
import random
import datetime


def get_image_list(image_dir):
    # TODO: we should split the dataset!
    image_list = [os.path.join(image_dir, i) for i in os.listdir(image_dir)]
    return image_list


def images2features(image_list, batch_size=32):
    crop_size = 178
    image_size = 128
    transform = [T.CenterCrop(crop_size), T.Resize(image_size), T.ToTensor(),
                 T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    transform = T.Compose(transform)
    extractor = bagnet.bagnet17(pretrained=True).cuda()
    extractor.eval()
    features = []

    def batch_generator():
        for i in range(0, len(image_list), batch_size):
            yield image_list[i:i + batch_size]

    for batch_image_list in tqdm(batch_generator(), total=math.ceil(len(image_list) / batch_size)):
        tensors = torch.Tensor(len(batch_image_list), 3, image_size, image_size)
        for i in range(len(batch_image_list)):
            image = Image.open(batch_image_list[i])
            image = transform(image)
            tensors[i] = image
        # the input shape should be (N,C,H,W)
        images = extractor(tensors.cuda())
        features.extend(images.cpu().detach().numpy())
        del tensors, images
    features = np.array(features)
    return features


def cluster_features(features, n_clusters, random_state=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(features)
    print(f"loss: {kmeans.inertia_}")
    return kmeans.cluster_centers_, kmeans.labels_


def main(config):
    print("start at: "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if config.debug:
        print("Running in debug mode!")
    cudnn.benchmark = True
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    images_list = get_image_list(config.dataset_path)
    if config.debug:
        images_list = random.sample(images_list, 100)
    print(f"images num: {len(images_list)}")
    features = images2features(images_list, config.batch_size)
    np.savez(config.save_path + "/features.npz", features)
    print("all images processed")
    print("start clustering")
    centers, labels = cluster_features(features, config.num_cluster)
    np.savez(config.save_path + "/clusters.npz", centers=centers, labels=labels)
    print("end at: "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--dataset_path', type=str, default='./data/celeba/')
    parser.add_argument('--num_cluster', type=int, default=40, help='number of cluster')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='./data/celeba/generated')
    parser.add_argument('--debug', type=bool, default=False)
    cfg = parser.parse_args()
    print(cfg)
    main(cfg)
