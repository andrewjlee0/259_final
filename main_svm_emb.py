import matplotlib.pyplot as plt
import time
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import pickle
import os

from utils_emb import *


def main(model_name, predictor_problem, dataset_name, test_problems, test_problems_name):

    # Get results of SVM decision boundary
    svm_output = do_svm(
        model_name, predictor_problem, dataset_name, 
        test_problems, test_problems_name
    )
    save_svm_output(
        svm_output, model_name, predictor_problem, 
        dataset_name, test_problems_name, mds=False
    )

    # Get results of SVM decision boundary for MDS reduced embeddings
    svm_output = do_svm_mds(
        model_name, predictor_problem, dataset_name, 
        test_problems, test_problems_name
    )
    save_svm_output(
        svm_output, model_name, predictor_problem, 
        dataset_name, test_problems_name, mds=True
    )



# Determine family of problems for SVM
# First problem in list is predictor problem
# SVM trains on predictor problem and evaluates on remaining problems
sd_problems = {
    'predictor_problem': 16,
    'test_problems': [15, 5, 22, 1, 19, 21, 7, 20],
    'all': [16, 15, 5, 22, 1, 19, 21, 7, 20]
}
io_problems = {
    'predictor_problem': 4,
    'test_problems': [2, 23],
    'all': [4, 2, 23]
}
io_problems_2 = {
    'predictor_problem': 23,
    'test_problems': [2, 4],
    'all': [23, 2, 4]
}
io_problems_3 = {
    'predictor_problem': 2,
    'test_problems': [4, 23],
    'all': [23, 2, 4]
}

# # Can SVM reproduce overall model accuracy using hidden layer embedding?
# for i in range(1, 24):
#     if i in sd_problems['all']:
#         continue

#     model_name = 'resnet18'
#     predictor_problem = i
#     dataset_name = 'svrt'
#     test_problems = i
#     test_problems_name = None
#     main(model_name, predictor_problem, dataset_name, test_problems, test_problems_name)


# # Can SVM reproduce overall model accuracy using hidden layer embeddings, for PSVRT dataset too?
# for i in sd_problems['all']:
#     model_name = 'resnet18'
#     predictor_problem = i
#     dataset_name = 'psvrt'
#     test_problems = i
#     test_problems_name = None
#     main(model_name, predictor_problem, dataset_name, test_problems, test_problems_name)


# # Can resnet trained on same-diff SVRT problem 16, which has highest test acc 
# # in Messina et al. (2021), generalize to other same-diff SVRT problems?
# model_name = 'resnet18'
# predictor_problem = sd_problems['predictor_problem']
# dataset_name = 'svrt'
# test_problems = sd_problems['test_problems']
# test_problems_name = 'sd'
# main(model_name, predictor_problem, dataset_name, test_problems, test_problems_name)


# # Can resnet trained on inside-outside SVRT problem 4 generalize to other 
# # inside-outside SVRT problems, which are conjunctive relations 
# # of inside-outside, building on top of the inside-outside relation?
# model_name = 'resnet18'
# predictor_problem = io_problems['predictor_problem']
# dataset_name = 'svrt'
# test_problems = io_problems['test_problems']
# test_problems_name = 'io'
# main(model_name, predictor_problem, dataset_name, test_problems, test_problems_name)


# # Can resnet trained on inside-outside SVRT problem 23 generalize to other 
# # inside-outside SVRT problems, which are simpler relations 
# # of inside-outside?
# model_name = 'resnet18'
# predictor_problem = io_problems_2['predictor_problem']
# dataset_name = 'svrt'
# test_problems = io_problems_2['test_problems']
# test_problems_name = 'io2'
# main(model_name, predictor_problem, dataset_name, test_problems, test_problems_name)


# Can resnet trained on inside-outside SVRT problem 2 generalize to other 
# inside-outside SVRT problems?
model_name = 'resnet18'
predictor_problem = io_problems_3['predictor_problem']
dataset_name = 'svrt'
test_problems = io_problems_3['test_problems']
test_problems_name = 'io3'
main(model_name, predictor_problem, dataset_name, test_problems, test_problems_name)