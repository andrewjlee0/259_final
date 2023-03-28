import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import time
from torchvision import models
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
import numpy as np
import os

from utils_train_test import *


def resnet18(pretrained=False):
    '''Returns untrained ResNet-18 with fully connected layer of 2 outputs (Cateogry 0, Category 1)
    '''

    # Initialize model, with untrained weights
    model = models.resnet18(pretrained=pretrained)

    # Get size of input layer
    num_ftrs = model.fc.in_features

    # Change final layer to have 2 outputs (Category 0, Category 1)
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Set to extract features of the second-to-last/penultimate layer: avgpool (n_dim=512)  
    model.avgpool.register_forward_hook(get_features('penult'))

    return model


def alexnet():
    '''Returns untrained AlexNet with fully connected layer of 2 outputs (Cateogry 0, Category 1)
    '''

    # Initialize model, with untrained weights
    model = models.alexnet()

    # Get size of penultimate linear layer
    num_ftrs_penult_lin = model.classifier[4].in_features

    # Change penultimate linear layer to output 512, not 4096, which is too many & can overfit
    model.classifier[4] = nn.Linear(num_ftrs_penult_lin, 512)

    # Get size of final linear layer
    num_ftrs_final_lin = model.classifier[4].out_features

    # Add new fully connected layer to final layer, with 2 outputs (Category 0, Category 1)
    model.classifier[6] = nn.Linear(num_ftrs_final_lin, 2)
    
    # Set to extract features of the second-to-last/penultimate layer: ReLu after classifer 4 (n_dim=512)  
    model.classifier[5].register_forward_hook(get_features('penult'))

    return model


def vit():
    return None

