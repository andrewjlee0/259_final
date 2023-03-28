import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import os
from math import floor
from torch.utils.data import DataLoader

from PSVRT.experiments.cnn.params import get_params
from PSVRT.instances.psvrt import psvrt


cudnn.benchmark = True
plt.ion()   # interactive mode


class Dataset(torch.utils.data.Dataset):
    'An SVRT dataset for PyTorch'

    def __init__(self, list_IDs, labels, transform, problem, img_dir):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.transform = transform
        self.problem = problem
        self.img_dir = img_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one image'
        # Select sample
        fname = self.list_IDs[index]

        # Load image and get label
        img_path = os.path.join(
            self.img_dir, 'results_problem_' + str(self.problem), fname)
        image = read_image(img_path)
        label = self.labels[fname]

        # Transform image (resize with bilinear interpolation, and normalize with 0.5 mean & std)
        image = self.transform(image)

        return image, label


class PSVRTDataset(torch.utils.data.Dataset):
    'A PSVRT dataset for PyTorch'

    def __init__(self, n_images, transform):
        'Initialization'
        self.n_images = n_images
        self.transform = transform

        params = get_params()
        self.raw_input_size = params['raw_input_size']
        self.train_data_init_args = params['train_data_init_args']

        self.psvrt = psvrt(self.raw_input_size, self.n_images)
        self.psvrt.initialize_vars(**self.train_data_init_args)

        self.images, self.labels = self.make_images_and_labels(
            self.psvrt, self.n_images, self.raw_input_size)

    def make_images_and_labels(self, psvrt, n_images, raw_input_size):
        'Create a dataset of n images with labels'

        # Create a dataset of images with their labels
        images, labels, _, _ = psvrt.single_batch()

        # Reshape variables for easier interpretation and programming
        images = images.reshape(n_images, raw_input_size[0], raw_input_size[1])
        labels = labels.reshape(n_images, 2)

        # Get only the second column of 'labels' matrix. This col has the relevant labels
        labels = labels[:,1]

        # Turn labels into a list
        labels = labels.reshape(self.n_images).tolist()

        return images, labels

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_images

    def __getitem__(self, index):
        'Generates one image'

        # Get an image and its label
        image = self.images[index]
        label = self.labels[index]

        # Convert image into PyTorch Tensor
        image = torch.from_numpy(image)

        # Turn grayscale (1D) to RGB (3D, as required by AlexNet)
        image.unsqueeze_(0)
        image = image.repeat(3, 1, 1)

        # Transform image (resize with bilinear interpolation, and normalize with 0.5 mean & std)
        image = self.transform(image)

        return image, label


def get_svrt_dataloaders(problem: int, ntrain: int, nval: int, ntest: int, batch_size: int, img_dir: str) -> dict[str, torch.utils.data.DataLoader]:
    '''Generates dictionary containing three DataLoader objects for train, val, test Datasets.
       Returns: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    '''

    # Get datasets from information passed into arguments
    data = get_svrt_datasets(
        problem, ntrain, nval, ntest, img_dir)

    # Create train, val, test dataloaders
    train_loader = DataLoader(
        data['train'], batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True
    )
    val_loader = DataLoader(
        data['val'], batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )
    test_loader = DataLoader(
        data['test'], batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )

    # Return dataloaders in a dictionary of dataloaders
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


def get_svrt_datasets(problem: int, ntrain: int, nval: int, ntest: int, img_dir: str, debug: bool = False) -> dict[str, Dataset]:
    '''Generates dictionary containing three custom Dataset objects, one for train, val, test.
       Returns: {'train': Dataset, 'val': Dataset, 'test': Dataset}
    '''

    # Necessary transform object: resize and normalize all images (train, val, test)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Format the image filenames and their respective category labels, amenable for Datasets object
    partition = get_IDs(ntrain, nval, ntest)
    labels = get_labels(ntrain, nval, ntest)
    if debug:
        print(partition)
        print(labels)

    # Generate train, val, and test datasets
    train_data = Dataset(
        partition['train'], labels, transform, problem, img_dir
    )
    val_data = Dataset(
        partition['val'], labels, transform, problem, img_dir
    )
    test_data = Dataset(
        partition['test'], labels, transform, problem, img_dir
    )

    # Return datasets in a dictionary of datasets
    return {'train': train_data, 'val': val_data, 'test': test_data}


def get_IDs(ntrain: int, nval: int, ntest: int, debug: bool = False) -> dict[str, list]:
    '''Generates partition variable, or dict where keys=train, val, test, values=list of filenames.
       Output is for Dataset object, which is loaded into Dataloader
    '''

    # Train set is first 400,000 images, val set is next 100,000, test set is final 100,000
    train_stop = floor(ntrain/2)  # 200000
    val_start = floor(ntrain/2)  # 200000
    val_stop = floor(train_stop) + floor(nval/2)  # 250000
    test_start = floor(val_stop)  # 250000
    test_stop = floor(val_stop) + floor(ntest/2)  # 300000
    if debug:
        print('train_stop', train_stop)
        print('val_start', val_start)
        print('val_stop', val_stop)
        print('test_start', test_start)
        print('test_stop', test_stop)

    # Initialize partition holder
    partition = {
        'train': None,
        'val': None,
        'test': None
    }

    # Create train set of equal numbers of pos & neg examples, each 200,000 images, and append to partition variable
    neg = ['sample_0_' +
           str(x) + '.png' if x > 999 else f'sample_0_{x:04}.png' for x in list(range(train_stop))]
    pos = ['sample_1_' +
           str(x) + '.png' if x > 999 else f'sample_1_{x:04}.png' for x in list(range(train_stop))]
    train = neg + pos
    partition['train'] = train

    # Create val set of equal numbers of pos & neg examples, each 50,000 images, and append to partition variable
    neg = ['sample_0_' +
           str(x) + '.png' if x > 999 else f'sample_0_{x:04}.png' for x in list(range(val_start, val_stop))]
    pos = ['sample_1_' +
           str(x) + '.png' if x > 999 else f'sample_1_{x:04}.png' for x in list(range(val_start, val_stop))]
    val = neg + pos
    partition['val'] = val

    # Create test set of equal numbers of pos & neg examples, each 50,000 images, and append to partition variable
    neg = ['sample_0_' +
           str(x) + '.png' if x > 999 else f'sample_0_{x:04}.png' for x in list(range(test_start, test_stop))]
    pos = ['sample_1_' +
           str(x) + '.png' if x > 999 else f'sample_1_{x:04}.png' for x in list(range(test_start, test_stop))]
    test = neg + pos
    partition['test'] = test

    return partition


def get_labels(ntrain: int, nval: int, ntest: int) -> dict[str, int]:
    '''Generates labels variable, or dict where keys=filenames, values=category (0,1)
    '''

    # Number of negative or positive images (half of all images)
    nhalf = floor((ntrain + nval + ntest) / 2)

    # Generate labels holder for positive and negative images
    neg = {
        'sample_0_' + str(x) + '.png' if x > 999 else f'sample_0_{x:04}.png':
        0 for x in list(range(nhalf))
    }  # example: {'sample_0_xxxx.png':0, 'sample_0_xxxx.png':0, etc.}
    pos = {
        'sample_1_' + str(x) + '.png' if x > 999 else f'sample_1_{x:04}.png':
        1 for x in list(range(nhalf))
    }  # example: {'sample_1_xxxx.png':1, 'sample_1_xxxx.png':1, etc.}

    # Combine neg and pos dictionaries
    labels = neg | pos
    return labels


def get_psvrt_datasets(n_images: int) -> dict[str, PSVRTDataset]:
    '''Generates dictionary containing a PSVRTDataset.
       Returns: {'test': PSVRTDataset}
    '''

    # Necessary transform object: resize and normalize all images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create PSVRT dataset
    test_data = PSVRTDataset(n_images, transform)

    # Return dictionary containing 'test' as only label
    # Not truly necessary, but necessary for idiosyncrasies of this code structure
    return {'test': test_data}


def get_psvrt_dataloaders(n_images: int, batch_size: int) -> dict[str, torch.utils.data.DataLoader]:
    '''Generates dictionary containing a DataLoader object for a PSVRTDataset.
       Returns: {'test': DataLoader}
    '''

    # Get PSVRT dataset with n images
    data = get_psvrt_datasets(n_images)

    # Create test dataloader
    test_loader = DataLoader(
        data['test'], batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )

    # Return dataloader in a dictionary of that single dataloader
    return {'test': test_loader}


# # SVRT problem
# problem = 1

# # Number of images for train, validation, test 
# ntrain, nval, ntest = 80, 10, 10

# # Batch size for dataloaders
# batch_size = 4

# # SVRT image directory
# img_dir = '/home/andrewlee0/svrt'

# get_dataloaders(problem, ntrain, nval, ntest, batch_size, img_dir)
