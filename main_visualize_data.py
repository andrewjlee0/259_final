import os
import matplotlib.pyplot as plt
import numpy as np

import torchvision

import torch
from utils_data import Dataset
from utils_data import get_svrt_datasets, get_psvrt_datasets


def imshow(inp: torch.Tensor, title: str = None) -> None:
    '''Display and save batch of images from Datasets (not Dataloaders)
    '''

    # Reorganize dimensions from (3, 224, 224) to (224, 224, 3)
    inp = inp.numpy().transpose((1, 2, 0))

    # Shift range of values from (-1, 1) to (0, 1)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean

    # Not sure what this does
    inp = np.clip(inp, 0, 1)

    # Display image and title
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

    # Save image to this 'visualize_data' directory
    plt.savefig('visualize_data/visualize_data.png')


def visualize_data(data: Dataset, batch_size: int) -> None:
    '''Displays a batch of images from Datasets of preprocess.py
       Would be easier to use Dataloaders, but not working right now
       due to batch size memory error on CPU?
    '''

    # Get a batch of training data
    images = []
    labels = []
    for i in list(range(batch_size)):

        # Get a single image and its associated category label
        image, label = data['train'][i]

        # Append image and label to the "batch" (lists)
        images.append(image)
        labels.append(label)

    # Make a grid from batch (grid is tensor object)
    grid = torchvision.utils.make_grid(images)

    # Save image to test.png in current directory ('visualize_data' folder)
    imshow(grid, title=['Category ' + str(x) for x in labels])


problem = 1
ntrain, nval, ntest = 80, 10, 10
img_dir = '/home/andrewlee0/svrt'
svrt_dataset = get_svrt_datasets(problem, ntrain, nval, ntest, img_dir)

n_images = 11200
data = get_psvrt_datasets(n_images)
psvrt_dataset = get_psvrt_datasets(n_images)

batch_size = 4
visualize_data(psvrt_dataset, batch_size)