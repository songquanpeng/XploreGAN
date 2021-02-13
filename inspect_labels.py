import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse

cluster = None
images_path = None
output_path = None


def load_image(index):
    path = f"{images_path}/{str(index + 1).zfill(6)}.jpg"
    return Image.open(path)


def show_images(label, n=5, display=False):
    num = n * n
    images = []
    for i in range(len(labels)):
        if len(images) >= num:
            break
        if label == labels[i]:
            images.append(load_image(i))
    fig, ax_arr = plt.subplots(n, n)
    fig.suptitle(f'Cluster {label}')
    fig.tight_layout()
    fig.set_size_inches(6, 8)
    i = 0
    for row in range(n):
        for col in range(n):
            if i < len(images):
                ax_arr[row, col].imshow(images[i])
                ax_arr[row, col].axis('off')
                i += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    if display:
        plt.show()
    plt.savefig(output_path + f"/cluster-{label}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_path', type=str, default='./data/celeba/generated/clusters.npz')
    parser.add_argument('--images_path', type=str, default=r"D:\Research\Data\celeba\images")
    parser.add_argument('--output_path', type=str, default="./data/celeba/generated")
    parser.add_argument('--n', type=int, default=5)
    cfg = parser.parse_args()
    print(cfg)
    images_path = cfg.images_path
    output_path = cfg.output_path
    cluster = np.load(cfg.cluster_path)
    labels = cluster["labels"]
    for i in tqdm(range(40)):
        show_images(i, cfg.n)
