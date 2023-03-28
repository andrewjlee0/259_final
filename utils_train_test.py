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

from utils_data import *


# Set up location for model training, specifically on GPU or CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##########################################################################################
# Function to get features/activations of a layer
##########################################################################################
# Intermediate placeholder for features (contents will continuously change every new image)
features = {}

def get_features(layer_name):
    '''Helper function for feature extraction of some layer (called 'module' for PyTorch)
       Key line to allow feature extraction of a layer is in model functions above: 
       'model.<insert layer name>.register_forward_hook( hook)'
    '''

    def hook(module, input, output):
        features[layer_name] = output
    return hook


##########################################################################################
# Train and test
##########################################################################################
def do_train_test(model, dataloaders: dict[str, DataLoader], dataset_sizes: dict[str, int], criterion, optimizer, scheduler, num_epochs: int, problem: int) -> tuple:
    '''Trains and tests a model on one SVRT problem.
       Returns tuple: (trained model with best parameters, train+test loss history, and train+test accuracy history)
    '''

    # Get all function arguments in a dictionary (this must be first line)
    # Because I'm lazy to type out all the function arguments again for train()
    args = locals()

    # Train model
    trained_model, y_loss, y_acc = train(**args)

    # Test model
    test_loss, test_acc, FEATS, LABELS = test(
        model, dataloaders, dataset_sizes, criterion, problem
    )

    # Update loss and acc history with test set loss and acc
    y_loss['test'] = [test_loss]
    y_acc['test'] = [test_acc]

    # y_loss will now look like this: {'train': [x,x,x,...], 'val': [x,x,x,...], 'test': [x]}
    # y_acc will also look like this: {'train': [x,x,x,...], 'val': [x,x,x,...], 'test': [x]}

    # Return best model, all loss history, all acc history, test feature activations, and test labels
    return model, y_loss, y_acc, FEATS, LABELS


def do_test(model, dataloaders: dict[str, DataLoader], dataset_sizes: dict[str, int], criterion, problem: int) -> tuple:
    '''Tests a model on just a test dataset.
       Returns tuple: (trained model with best parameters, test loss, and test accuracy)
    '''

    # Test model
    test_loss, test_acc, FEATS, LABELS = test(
        model, dataloaders, dataset_sizes, criterion, problem
    )

    # Update loss and acc history with test set loss and acc
    y_loss = {'test': [test_loss]}
    y_acc = {'test': [test_acc]}

    # y_loss will now look like this: {'test': [x]}
    # y_acc will also look like this: {'test': [x]}

    # Return all loss history, all acc history, test feature activations, and test labels
    return y_loss, y_acc, FEATS, LABELS


def train(model, dataloaders: dict[str, DataLoader], dataset_sizes: dict[str, int], criterion, optimizer, scheduler, num_epochs: int, problem: int):
    '''Returns tuple: (trained model with best performing parameters, training loss history, and training accuracy history)
    '''

    # Get current time
    since = time.time()

    # Initialize iteratively updated variables
    best_model_wts = copy.deepcopy(model.state_dict()) # current best weights (should be random)
    best_acc = 0.0 # current best accuracy
    y_loss = {'train': [], 'val': []} # empty loss history
    y_acc = {'train': [], 'val': []} # empty acc history

    # Iterate over epochs
    for epoch in range(num_epochs):

        # Print current epoch
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # FEATS = []
        # LABELS = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Current number of correct predictions
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # Final layer activations: Tensor([x1, x2])
                    _, preds = torch.max(outputs, 1) # Tuple: (Maximum of final layer activations: e.g. Tensor(x2), Index of max)
                    loss = criterion(outputs, labels)
                    # Clarification: 'preds' variable is either 0,1 because it is set to the index of the maximum. Coincidentally, 
                    # the labels of the SVRT categories are also 0,1. Thus, the indices are used as the "predictions" of the model,
                    # even though the final layer output is not a binary 0,1 integer.

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #     # Append penultimate features to FEATS, and final layer predictions (0 or 1) with labels to PREDS
            #     _feats = features['penult'].detach().cpu().numpy()
            #     print(outputs.detach().cpu().numpy())
            #     FEATS.append(_feats)
            
            # print('this is feats:', FEATS)

            # I don't know what this does exactly, but something related to learning rate
            if phase == 'train':
                scheduler.step()

            # Calculate and print loss and acc for this epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Append to list of losses
            y_loss[phase].append(epoch_loss)
            y_acc[phase].append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # Separate terminal print output for every epoch
        print()

    # Print how long training took
    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return best model, loss history, acc history
    return model, y_loss, y_acc


def test(model, dataloaders: dict[str, DataLoader], dataset_sizes: dict[str, int], criterion, problem: int) -> tuple:
    '''Predicts on test set of SVRT images and extracts features of layers of interest.
       Returns tuple: (test set loss, test set accuracy, and feature extractions)
    '''

    # Get current time
    since = time.time()

    # Set model to evaluate mode
    model.eval()

    # LABELS is storage of actual labels of ALL TEST ITEMS, split into batches
    # FEATS is storage of all desired layers' features/activations of ALL TEST ITEMS
    # Each item in a list of FEATS is a feature vector (python list, not numpy array)
    LABELS = []
    FEATS = {'final': [], 'penult': []}

    # Current number of correct predictions
    running_loss = 0.0
    running_corrects = 0

    # Iterate through every example in test dataset
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # I don't know why, but there was this Runtime Error for the PSVRT case, 
        # and only after looking online did this following line work. 
        labels = labels.to(torch.int64)

        # Forward
        torch.set_grad_enabled(False)
        outputs = model(inputs) # Final layer activations: Tensor([x1, x2])
        _, preds = torch.max(outputs, 1) # Tuple: (Maximum of final layer activations: e.g. Tensor(x2), Index of max)
        loss = criterion(outputs, labels)
        # Clarification: 'preds' variable is either 0,1 because it is set to the index of the maximum. Coincidentally, 
        # the labels of the SVRT categories are also 0,1. Thus, the indices are used as the "predictions" of the model,
        # even though the final layer output is not a binary 0,1 integer.

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # Format variables to lists for saving
        labels = labels.detach().cpu().numpy().tolist()
        penult = features['penult'].detach().cpu().numpy()
        final = outputs.detach().cpu().numpy().tolist()

        # Reduce dimensionality of 'penult' variable as it is unnecessariily large
        # Goes from dxdxdxd... to dxd (2 dimensions)
        penult = penult.reshape(*list(penult.shape)[:2]).tolist()

        # Append final layer and penultimate activations to FEATS and labels to LABELS
        LABELS += labels
        FEATS['penult'] += penult
        FEATS['final'] += final

    # Calculate and print loss and acc for this epoch
    test_loss = running_loss / dataset_sizes['test']
    test_acc = (running_corrects.double() / dataset_sizes['test']).item()
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # Print how long training took
    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Return loss history, acc history, test feature activations, test labels
    return test_loss, test_acc, FEATS, LABELS


##########################################################################################
# Save training and test output
##########################################################################################
def save_history(model_name: str, problem: int, y_loss: dict[str, list], y_acc: dict[str, list], dataset_name: str, test_problem: int = None) -> None:
    '''Saves a model's loss and accuracy history: {'train': [x,x,x,...], 'val': [x,x,x,...], 'test': [x]}
    '''

    # Create folder for model output, if doesn't exist
    folder = os.path.join('models', model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename ending
    if test_problem is not None:
        ending = dataset_name + '_' + str(test_problem)
    else:
        ending = dataset_name

    # Filenames
    loss_fname = str(problem) + '_loss_' + model_name + '_' + ending + '.pickle'
    acc_fname = str(problem) + '_acc_' + model_name + '_' + ending + '.pickle'
    PATH_loss = os.path.join(folder, loss_fname)
    PATH_acc = os.path.join(folder, acc_fname)

    # Save loss history 
    with open(PATH_loss, 'wb') as PATH_loss:
        pickle.dump(y_loss, PATH_loss, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save accuracy history
    with open(PATH_acc, 'wb') as PATH_acc:
        pickle.dump(y_acc, PATH_acc, protocol=pickle.HIGHEST_PROTOCOL)


def save_features(model_name: str, predictor_problem: int, feats: dict, labels: list, dataset_name: str, test_problem: int = None) -> None:
    '''Saves a model's layer features/activations.
    '''

    # Create folder for model output, if doesn't exist
    folder = os.path.join('models', model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename ending
    if test_problem is not None:
        ending = dataset_name + '_' + str(test_problem)
    else:
        ending = dataset_name
    
    # Filenames
    feats_fname = str(predictor_problem) + '_test_feats_' + model_name + '_' + ending + '.pickle'
    labels_fname = str(predictor_problem) + '_test_labels_' + model_name + '_' + ending + '.pickle'
    PATH_feats = os.path.join(folder, feats_fname)
    PATH_labels = os.path.join(folder, labels_fname)

    # Save features
    with open(PATH_feats, 'wb') as PATH_feats:
        pickle.dump(feats, PATH_feats, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save predictions
    with open(PATH_labels, 'wb') as PATH_labels:
        pickle.dump(labels, PATH_labels, protocol=pickle.HIGHEST_PROTOCOL)

