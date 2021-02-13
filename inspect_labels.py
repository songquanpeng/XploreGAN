import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

cluster = np.load("./data/celeba/generated/clusters.npz")
labels = cluster["labels"]
images_path = r"D:\Research\Data\celeba\images"
output_path = "./data/celeba/generated"


def load_image(index):
    path = f"{images_path}/{str(index + 1).zfill(6)}.jpg"
    return Image.open(path)


def show_images(label, display=False, n=5):
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
    for label in tqdm(range(40)):
        show_images(label)
