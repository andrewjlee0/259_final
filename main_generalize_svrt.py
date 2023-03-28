import matplotlib.pyplot as plt
import time
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
import os
from sklearn import svm

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
# 
##########################################################################################
# Input parameters
model_name = 'resnet18'
predictor_problem = 23
test_problems = [4, 2]
# predictor_problem = 16
# test_problems = [15, 5, 22, 1, 19, 21, 7, 20]

# Path for best weights of predictor problem
fname = str(predictor_problem) + '_' + model_name + '.pth'
PATH = os.path.join('models', model_name, fname)

# Load best model of predictor problem
model = resnet18()
model.load_state_dict(torch.load(PATH))
model.eval()

# Move model to GPU
model = model.to(DEVICE)


##########################################################################################
# 
##########################################################################################
for test_problem in test_problems:

    print('Test problem:', test_problem)
    print()

    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    batch_size = 64


    ##########################################################################################
    # Data
    ##########################################################################################
    # Number of SVRT test images
    ntrain, nval, ntest = 2, 2, 11200

    # Folder containing SVRT images
    img_dir = '/home/andrewlee0/svrt'
    
    # Get train, val, test Datasets (all stored inside 'data' variable)
    data = get_svrt_datasets(
        test_problem, ntrain, nval, ntest, img_dir
    )

    # Get the length of each of the Datasets
    dataset_sizes = {'test': len(data['test'])}

    # Get train, val, test DataLoaders
    dataloaders = get_svrt_dataloaders(
        test_problem, ntrain, nval, ntest, batch_size, img_dir
    )


    ##########################################################################################
    # 
    ##########################################################################################
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Train/test the model
    y_loss, y_acc, FEATS, LABELS  = do_test(
        model, dataloaders, dataset_sizes, criterion, test_problem
    )


    ##########################################################################################
    # Saving
    ##########################################################################################
    # Save loss and acc histories
    save_history(model_name, predictor_problem, y_loss, y_acc, 'svrt', test_problem = test_problem)

    # Save layer features/activations
    save_features(model_name, predictor_problem, FEATS, LABELS, 'svrt', test_problem = test_problem)

    print()