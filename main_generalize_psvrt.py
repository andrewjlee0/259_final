import matplotlib.pyplot as plt
import time
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
import os

from utils_data import *
from utils_models import *
from utils_train_test import *


##########################################################################################
# GPU setup
##########################################################################################
# Set up location for model training, specifically on GPU or CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# I don't know what this does
cudnn.benchmark = True
plt.ion()   # interactive mode

dataset_name = 'svrt'

##########################################################################################
# Test various models on the PSVRT same-different problem 
##########################################################################################
for problem in range(1, 24):

    print('Problem:', problem)
    print()

    ##########################################################################################
    # Initialize all models
    ##########################################################################################
    # ResNet-18 with untrained weights and final layer of 2 nodes
    all_models = [
        # ('resnet18', resnet18()),
        ('alexnet', alexnet()),
        # ('transformer', vit())
    ]

    ##########################################################################################
    # Iteratively test all models
    ##########################################################################################   
    for model_name, model in all_models:

        print('Model:', model_name)
        print()


        ##########################################################################################
        # Hyperparameters
        ##########################################################################################
        batch_size = 64


        ##########################################################################################
        # Data
        ##########################################################################################
        # Number of PSVRT test images
        n_images = 11200
        
        # Get train, val, test Datasets (all stored inside 'data' variable)
        data = get_psvrt_datasets(n_images)

        # Get the length of each of the Datasets
        dataset_sizes = {'test': len(data['test'])}

        # Get train, val, test DataLoaders
        dataloaders = get_psvrt_dataloaders(n_images, batch_size)


        ##########################################################################################
        # Test on PSVRT
        ##########################################################################################
        # Load the best model from SVRT training session
        PATH = os.path.join('models', model_name, str(problem) + '_' + model_name + '.pth')
        model.load_state_dict(torch.load(PATH))
        model.eval()
        
        # Move model to GPU
        model = model.to(DEVICE)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Train/test the model
        y_loss, y_acc, FEATS, LABELS  = test(
            model, dataloaders, dataset_sizes, criterion, problem
        )


        ##########################################################################################
        # Saving
        ##########################################################################################
        # Save loss and acc histories
        save_history(model_name, problem, y_loss, y_acc, dataset_name)

        # Save layer features/activations
        save_features(model_name, problem, FEATS, LABELS, dataset_name)

        print()