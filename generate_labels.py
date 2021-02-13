import os
import argparse
import numpy as np
import torch
from torch.backends import cudnn
from pretrained import bagnet
from torchvision import transforms as T
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
import math
import random
import time


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


def generate_features(config):
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    images_list = get_image_list(config.dataset_path)
    if config.debug:
        images_list = random.sample(images_list, 100)
    print(f"Images num: {len(images_list)}")
    features = images2features(images_list, config.batch_size)
    return features


def process_features(features, pca_dim=256):
    # L2 normalize
    features = normalize(features, norm='l2')
    # PCA
    pca = PCA(n_components=pca_dim)
    features = pca.fit_transform(features)
    return features


def cluster_features(features, n_clusters, random_state=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=3000)
    kmeans.fit(features)
    print(f"Loss: {kmeans.inertia_}")
    return kmeans.cluster_centers_, kmeans.labels_


def main(config):
    start_time = time.time()
    if config.debug:
        print("Running in debug mode.")
    cudnn.benchmark = True
    if config.features_generated:
        features = np.load(config.save_path + "/features.npz")["arr_0"]
    else:
        print("Start generating.")
        features = generate_features(config)
        np.savez(config.save_path + "/features.npz", features)
    print("Features generated.")

    if config.features_processed:
        processed_features = np.load(config.save_path + "/processed_features.npz")["arr_0"]
    else:
        print("Start processing.")
        processed_features = process_features(features, config.pca_dim)
        np.savez(config.save_path + "/processed_features.npz", processed_features)
    print("Features processed.")

    if config.features_clustered:
        clusters = np.load(config.save_path + "/clusters.npz")
        centers = clusters["centers"]
        labels = clusters["labels"]
    else:
        print("Start clustering.")
        centers, labels = cluster_features(processed_features, config.num_cluster)
    print("Features clustered.")

    clusters = [[] for _ in range(config.num_cluster)]
    for i in range(len(labels)):
        label = labels[i]
        clusters[label].append(i)
    stds = np.zeros((config.num_cluster,))
    for i in range(config.num_cluster):
        cluster = features[clusters[i]]
        stds[i] = np.std(cluster)
    np.savez(config.save_path + "/clusters.npz", centers=centers, labels=labels, stds=stds)
    end_time = time.time()
    print(f"All done (about {(end_time - start_time) / 60:.2f} minutes used).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--dataset_path', type=str, default='./data/celeba/')
    parser.add_argument('--num_cluster', type=int, default=100, help='used in kmeans, 100 for 128x128, 50 for 256x256')
    parser.add_argument('--pca_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='./data/celeba/generated')
    parser.add_argument('--features_generated', type=bool, default=False)
    parser.add_argument('--features_processed', type=bool, default=False)
    parser.add_argument('--features_clustered', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    cfg = parser.parse_args()
    print(cfg)
    main(cfg)
