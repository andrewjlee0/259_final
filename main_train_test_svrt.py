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


##########################################################################################
# Train and test various models for each SVRT problem (23 total)
##########################################################################################
# for problem in range(1, 24):
for problem in [21]:

    print('Problem:', problem)
    print()

    ##########################################################################################
    # Initialize all models
    ##########################################################################################
    # ResNet-18 with untrained weights and final layer of 2 nodes
    all_models = [
        ('resnet18_pretrained', resnet18(pretrained=True)),
        ('resnet18', resnet18()),
        # ('alexnet', alexnet()),
        # ('transformer', vit())
    ]

    ##########################################################################################
    # Iteratively train/test/save all models
    ##########################################################################################   
    for model_name, model in all_models:

        print('Model:', model_name)
        print()

        ##########################################################################################
        # Hyperparameters
        ##########################################################################################
        num_epochs = 50
        batch_size = 64
        learning_rate = 0.001
        momentum = 0.9
        step_size = 5
        gamma = 0.1


        ##########################################################################################
        # Data
        ##########################################################################################
        # Train, val, test split
        ntrain, nval, ntest = 28000, 5600, 11200

        # Folder containing SVRT images
        img_dir = '/home/andrewlee0/svrt'
        
        # Get train, val, test Datasets (all stored inside 'data' variable)
        data = get_svrt_datasets(
            problem, ntrain, nval, ntest, img_dir
        )
        if False:
            print(data['test'][0][0])
            print(data['test'][0][0].shape)
            print(torch.sum(data['test'][0][0][0]) == torch.sum(data['test'][0][0][1]))
            print(torch.sum(data['test'][0][0][0]) == torch.sum(data['test'][0][0][2]))
            print(torch.sum(data['test'][0][0][1]) == torch.sum(data['test'][0][0][2]))

        # Get the length of each of the Datasets
        dataset_sizes = {x: len(data[x]) for x in ['train', 'val', 'test']}

        # Get train, val, test DataLoaders
        dataloaders = get_svrt_dataloaders(
            problem, ntrain, nval, ntest, batch_size, img_dir
        )


        ##########################################################################################
        # Train/Test on SVRT
        ##########################################################################################
        # Move model to GPU
        model = model.to(DEVICE)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        )

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        # Train/test the model
        trained_model, y_loss, y_acc, FEATS, LABELS = do_train_test(
            model, dataloaders, dataset_sizes, criterion, optimizer,
            exp_lr_scheduler, num_epochs, problem
        )


        ##########################################################################################
        # Saving
        ##########################################################################################
        # Get state dictionary, representing model's best parameters (not necessarily last epoch)
        state = trained_model.state_dict()

        # Save state dictionary to 'models' directory
        PATH = 'models/' + model_name + '/' + str(problem) + '_' + str(model_name) + '.pth'
        torch.save(state, PATH)

        # Save loss and acc histories
        save_history(model_name, problem, y_loss, y_acc, 'svrt')

        # Save layer features/activations
        save_features(model_name, problem, FEATS, LABELS, 'svrt')

        print()




