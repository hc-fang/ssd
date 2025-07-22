import sys
import os
import json

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from fedbox.utils.data import split_uniformly, split_dirichlet_label, split_by_label

from fedbox.utils.training import set_seed
from utils.plot import plot_label_distribution_each_client, plot_label_distribution

def download_tiny_imagenet(root):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_path = os.path.join(root, "tiny-imagenet-200")
    
    if not os.path.exists(dataset_path):
        print("Downloading Tiny ImageNet-200...")
        download_and_extract_archive(url, root)
    else:
        print("Tiny ImageNet-200 already exists.")

def main():
    set_seed(1)
    torchvision_root = './data'
    dataset_name = 'cifar100'
    client_num = 10
    class_per_client = 10
    alpha = 0.1
    split_method = 'dir' 

    if dataset_name == 'cifar10':
        dataset = CIFAR10(torchvision_root, download=True)
        n_classes = 10
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(torchvision_root, download=True)
        n_classes = 100
    elif dataset_name == 'tinyimagenet200':
        download_tiny_imagenet(torchvision_root)
        dataset_path = os.path.join(torchvision_root, "tiny-imagenet-200", "train")
        dataset = ImageFolder(root=dataset_path)
        n_classes = 200

    os.makedirs(os.path.join(torchvision_root, 'fed'), exist_ok=True)

    if split_method == 'dir':
        split_file = f'fed/{dataset_name}_dir_client{client_num}_alpha{alpha}.json'
        results = split_dirichlet_label(
            np.arange(len(dataset)),
            np.array(dataset.targets),
            client_num=client_num,
            alpha=alpha
        )
    elif split_method == 'pat':
        split_file = f'fed/{dataset_name}_path_client{client_num}_classpclient{class_per_client}.json'
        results = split_by_label(
            np.arange(len(dataset)), 
            np.array(dataset.targets), 
            client_num=client_num, 
            class_per_client=class_per_client
        )
    else:
        raise ValueError("unknown method to split dataset")

    label_distribution_each_client = [[0 for _ in range(n_classes)] for _ in range(client_num)]
    label_distribution = [0 for _ in range(n_classes)]

    for i, (ids, labels) in enumerate(results):
        for l in labels:
            label_distribution_each_client[i][l] += 1
            label_distribution[l] += 1

    plot_label_distribution_each_client(label_distribution_each_client, os.path.join('data', split_file.replace('json', 'png')))
    plot_label_distribution(label_distribution, os.path.join('data', split_file.replace('.json', '_all.png')))

    os.makedirs(os.path.join('data', 'fed'), exist_ok=True)
    with open(os.path.join('data', split_file), 'w') as json_file:
        json.dump([indices.tolist() for indices, _ in results], json_file, indent=4)


if __name__ == '__main__':
    main()
