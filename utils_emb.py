import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import time
from torchvision import models
import torch
import pickle
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Make plots look nice
sns.set_context('notebook')


##########################################################################################
# Dimensionality reduction
##########################################################################################
def normalize_features(feature_vectors):
    '''
    '''

    # Divide each value in vector by its respective vector sum
    sums = np.sum(feature_vectors, axis=1)
    normalized_vectors = feature_vectors / sums[:, None]

    return normalized_vectors


def get_mds(X, labels):
    '''
    '''

    # Put features and labels into compact DataFrame
    X_df = pd.DataFrame(data = X)
    X_df['labels'] = labels

    # Get all unique labels and number of unique labels
    unique_labels = X_df['labels'].unique()
    n_unique = len(unique_labels)

    # Can't do MDS on everything, so get subset of feature/label DataFrame
    n_samples = 1500
    reduced_X_df = pd.DataFrame()

    # Sample equal number of rows from dataset
    for label in unique_labels:
        samples_label = X_df[X_df.labels == label].sample( 
            int(n_samples / n_unique), random_state = 0)
        reduced_X_df = pd.concat([reduced_X_df, samples_label])

    # Split dataframe into features and labels separately
    reduced_X = reduced_X_df.loc[:, reduced_X_df.columns != 'labels'].to_numpy()
    reduced_labels = reduced_X_df['labels'].to_list()

    # Run MDS
    new_mds = MDS(random_state = 0, n_components = 3)
    X_mds = new_mds.fit_transform(reduced_X)

    # Put MDS results into dataframe, with labels as new column
    mds_df = pd.DataFrame(data = X_mds, columns = ['mds1', 'mds2', 'mds3'])
    mds_df['labels'] = reduced_labels

    # Return MDS results
    return mds_df


def get_lda(X, labels):
    '''
    '''

    # Run LDA
    new_lda = LinearDiscriminantAnalysis() # output num of LDA components is n_classes-1 = 4-1 = 3
    X_lda = new_lda.fit_transform(X, labels)

    # Return LDA results
    return X_lda, labels


##########################################################################################
# Plot embeddings
##########################################################################################
def do_plot_emb(model_name, predictor_problem, dataset_name, test_problems, test_problems_name, svm_output):
    '''Visualize the features/activations using a dimensionality reduction technique
    '''

    # Note: The embeddings in 'feats' and 'labels' are assumed to be lists, not numpy arrays
    # or pytorch tensors. The length of both lists should be the number of test items,
    # and an element in a list should be a test item embedding. The index of an element should be
    # the index of the corresponding test item.

    # Measure duration of dimensionality reduction procedure
    start = time.time()

    # For now, only visualized layer is penultimate layer; can implement more
    # For now, only dimensionality reduction is MDS
    layer_name = 'penult'
    method = 'mds'

    # Combine list of features and labels for all problems
    all_feats, all_labels, all_feats_labels = get_all_feats_labels(
        model_name, predictor_problem, dataset_name, 
        test_problems, test_problems_name, layer_name
    )

    # Normalize all features
    fvs_normalized = normalize_features(all_feats[layer_name])

    # Reduce dimensionality of feature vectors and plot it
    fig = get_mds_plot(
        fvs_normalized, all_labels, svm_output, dataset_name,
        test_problems_name
    )

    # Save plot
    save_plot_emb(
        fig, method, model_name, layer_name, 
        predictor_problem, dataset_name, test_problems_name
    )
    # print(predictor_problem, svm_output[layer_name]['test'])

    print('Duration: {} seconds'.format(time.time() - start))


def get_mds_plot(X, labels, svm_output, dataset_name, test_problems_name):
    '''
    '''

    # Get MDS results
    mds_df = get_mds(X, labels)
    n_unique = len(mds_df['labels'].unique())
    
    # Change labels based on number of labels
    if n_unique == 2:
        mds_df['labels'] = mds_df['labels'].map({
            0: 'Category 1', # light blue
            1: 'Category 2' # dark blue
        })

    elif dataset_name == 'psvrt':
        mds_df['labels'] = mds_df['labels'].map({
            0: 'Category 1',
            1: 'Category 2',
            2: 'PSVRT Diff',
            3: 'PSVRT Same'
        })

    elif test_problems_name in ['io','io2','io3']:
        if test_problems_name == 'io2':
            mds_df['labels'] = mds_df['labels'].map({
                0: 4,
                1: 5,
                2: 2,
                3: 3,
                4: 0,
                5: 1
            })
        elif test_problems_name == 'io3':
            print(mds_df)
            mds_df['labels'] = mds_df['labels'].map({
                0: 2,
                1: 3,
                2: 0,
                3: 1,
                4: 4,
                5: 5
            })

        # mds_df['labels'] = mds_df['labels'].map({
        #     0: 'P4 Outside', # light blue
        #     1: 'P4 Inside', # dark blue
        #     2: 'P2 Inside-near-center', # light green
        #     3: 'P2 Inside-near-border', # dark green
        #     4: 'P23 Two-outside or two-inside', # light red
        #     5: 'P23 One-outside and one-inside' # dark red
        # })

    # # Plot MDS results
    # fig = plt.figure(figsize = (6,6))
    # fig.tight_layout()
    # sns.scatterplot(
    #     data = mds_df,
    #     x = 'mds1',
    #     y = 'mds2',
    #     hue = 'labels',
    #     palette = sns.color_palette('Paired', n_colors = n_unique), 
    # ).set(
    #     xlabel = 'MDS dim 1',
    #     ylabel = 'MDS dim 2'
    # )
    # sns.despine()
    # plt.legend(title = '', bbox_to_anchor = (0.95, 1.0))

    # # Save ylim of this scatter plot before SVM lines for below reasons
    # original_ylim = plt.ylim()

    # # Plot Predictor SVM decision line
    # m = svm_output['predictor_svm_results']['m']
    # b = svm_output['predictor_svm_results']['b']
    # x_points = np.linspace( *plt.xlim())    # generating x-points from -1 to 1
    # y_points = -(m[0] / m[1]) * x_points - b / m[1]  # getting corresponding y-points
    # plt.plot(x_points, y_points, c='b')

    # # Plot rest of SVM lines
    # colors = ['green','red','orange','purple','yellow','blue']
    # for test_problem, color in zip(svm_output['test_problems'], colors):
    #     for dsname in svm_output[test_problem]:

    #         if test_problem == svm_output['predictor_problem'] and dsname == 'svrt':
    #             continue

    #         m = svm_output[test_problem][dsname]['m']
    #         b = svm_output[test_problem][dsname]['b']
    #         x_points = np.linspace( *plt.xlim())    # generating x-points from -1 to 1
    #         y_points = -(m[0] / m[1]) * x_points - b / m[1]  # getting corresponding y-points
    #         plt.plot(x_points, y_points, c=color)

    # # Addition of SVM lines will distort y-axis, so set it back to original scatter plot
    # plt.ylim( original_ylim)

    # return fig

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette('Paired').as_hex())

    # plot
    sc = ax.scatter(
        mds_df['mds1'], 
        mds_df['mds2'], 
        mds_df['mds3'], 
        c = mds_df['labels'], 
        marker='o', 
        cmap= plt.get_cmap('Paired', 6)
    )
    ax.set_xlabel('MDS dim 1')
    ax.set_ylabel('MDS dim 2')
    ax.set_zlabel('MDS dim 3')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1, 1), loc=2)


    # # Plot MDS results
    # fig, ax = plt.subplots(1,3, figsize = (12,6))
    # fig.tight_layout()
    # sns.scatterplot(
    #     data = mds_df,
    #     x = 'mds2',
    #     y = 'mds1',
    #     hue = 'labels',
    #     ax = ax[0],
    #     palette = sns.color_palette('Paired', n_colors = n_unique),
    # ).set(
    #     xlabel = 'MDS dim 2',
    #     ylabel = 'MDS dim 1'
    # )
    # ax[0].get_legend().remove()
    # sns.despine()

    # sns.scatterplot(
    #     data = mds_df,
    #     x = 'mds2',
    #     y = 'mds3',
    #     hue = 'labels',
    #     ax = ax[1],
    #     palette = sns.color_palette('Paired', n_colors = n_unique),
    # ).set(
    #     xlabel = 'MDS dim 2',
    #     ylabel = 'MDS dim 3'
    # )
    # ax[1].get_legend().remove()
    # sns.despine()

    # sns.scatterplot(
    #     data = mds_df,
    #     x = 'mds1',
    #     y = 'mds3',
    #     hue = 'labels',
    #     ax = ax[2],
    #     palette = sns.color_palette('Paired', n_colors = n_unique),
    # ).set(
    #     xlabel = 'MDS dim 1',
    #     ylabel = 'MDS dim 3'
    # )
    # ax[2].legend(title='Category', bbox_to_anchor = [1,1]).get_frame()
    # sns.despine()

    return fig


def save_plot_emb(fig, method, model_name, layer_name, predictor_problem, dataset_name, test_problems_name = None):
    '''
    '''

    # Just state the figure? 
    fig

    # Create folder, if doesn't exist
    folder = os.path.join('models', model_name, 'plot_emb')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename ending
    if test_problems_name is not None:
        ending = dataset_name + '_' + str(test_problems_name)
    else:
        ending = dataset_name
    print(ending)
    
    # Construct filename
    fname = str(predictor_problem) + '_' + layer_name + '_' + method + '_' + model_name + '_' + ending + '.png'
    PATH = os.path.join(folder, fname)

    # Save figure
    plt.savefig(PATH, bbox_inches="tight")


##########################################################################################
# SVM
##########################################################################################
def get_svm(fvs, labels):
    # Find SVM decision line that best separates the baseline embeddings
    clf = SVC(kernel='linear') # Linear Kernel
    clf.fit(fvs, labels)
    return clf


def do_svm(model_name, predictor_problem, dataset_name, test_problems, test_problems_name):
    '''
    '''

    start = time.time()
    test_problems = [test_problems] if type(test_problems) is not list else test_problems

    # Only concerned with penult layers; future can implement all 
    layer_name = 'penult'

    # Combine list of features and labels for all problems
    all_feats, all_labels, all_feats_labels = get_all_feats_labels(
        model_name, predictor_problem, dataset_name, 
        test_problems, test_problems_name, layer_name
    )

    # Normalize all features
    all_fvs_normalized = normalize_features(all_feats[layer_name])

    all_fvs_df = pd.DataFrame(data = all_fvs_normalized)
    all_fvs_df['labels'] = all_labels
    
    predictor_fvs_mds, predictor_labels_mds = decompose_df_feats_labels(
        all_fvs_df, unique_labels = [0,1]
    )

    # Find decision line of predictor problem 
    predictor_clf = get_svm(predictor_fvs_mds, predictor_labels_mds)

    # Summary of SVM fit onto baseline embeddings and predicted labels of test problems
    svm_output = {
        'predictor_problem': predictor_problem,
        'test_problems': test_problems,
        'predictor_svm_results': {
            'svm': predictor_clf,
            'm': predictor_clf.coef_[0],
            'b': predictor_clf.intercept_[0]
        }
    }

    for test_problem in test_problems:
        for ds_name in all_feats_labels[test_problem]:

            # 1. Find decision line of test problem MDS, and predict to test problem
            unique_adj_test_labels = set(all_feats_labels[test_problem][ds_name]['labels'])
            test_fvs_mds, test_labels_mds = decompose_df_feats_labels(
                all_fvs_df, unique_labels = unique_adj_test_labels
            )
            test_clf = get_svm(test_fvs_mds, test_labels_mds)
            test_labels_svm = test_clf.predict(test_fvs_mds)
            print(test_labels_svm)

            # Get accuracy across all SVM's labels and true labels for a test problem
            test_svm_acc = accuracy_score(test_labels_mds, test_labels_svm)


            # 2. Predict Predictor SVM to test problem
            predictor_labels_svm = predictor_clf.predict(test_fvs_mds)

            # Since Predictor SVM outputs are 0s and 1s, we have to adjust them back to the
            # adjusted test label values (2,3,4,5...)
            class0 = min(unique_adj_test_labels)
            class1 = max(unique_adj_test_labels)
            predictor_labels_svm = [class0 if l == 0 else class1 for l in predictor_labels_svm]

            # Get accuracy across all SVM's labels and true labels for a test problem
            predictor_svm_acc = accuracy_score(test_labels_mds, predictor_labels_svm)

            # Save SVM accuracy to summary output
            if test_problem not in svm_output:
                svm_output[test_problem] = {}
            svm_output[test_problem][ds_name] = {
                'svm': test_clf,
                'm': test_clf.coef_[0],
                'b': test_clf.intercept_[0],
                'test_svm_acc': test_svm_acc,
                'predictor_svm_acc':predictor_svm_acc
            }

    print('Duration: {} seconds'.format(time.time() - start))

    # Return summary output
    return svm_output


##########################################################################################
# SVM on MDS Embeddings
##########################################################################################
def do_svm_mds(model_name, predictor_problem, dataset_name, test_problems, test_problems_name) -> dict:
    '''
    '''

    start = time.time()
    test_problems = [test_problems] if type(test_problems) is not list else test_problems

    # Only concerned with penult layers; future can implement all 
    layer_name = 'penult'

    # Combine list of features and labels for all problems
    all_feats, all_labels, all_feats_labels = get_all_feats_labels(
        model_name, predictor_problem, dataset_name, 
        test_problems, test_problems_name, layer_name
    )

    # Normalize all feature vectors across all problems and do MDS together with all problems
    all_fvs_normalized = normalize_features(all_feats[layer_name])
    mds_df = get_mds(all_fvs_normalized, all_labels)

    # Find decision line of predictor problem MDS
    predictor_fvs_mds, predictor_labels_mds = decompose_df_feats_labels(
        mds_df, unique_labels = [0,1])
    predictor_clf = get_svm(predictor_fvs_mds, predictor_labels_mds)

    # Summary of SVM fit onto baseline embeddings and predicted labels of test problems
    svm_mds_output = {
        'predictor_problem': predictor_problem,
        'test_problems': test_problems,
        'predictor_svm_results': {
            'svm': predictor_clf,
            'm': predictor_clf.coef_[0],
            'b': predictor_clf.intercept_[0]
        }
    }

    for test_problem in test_problems:
        for ds_name in all_feats_labels[test_problem]:

            # 1. Find decision line of test problem MDS, and predict to test problem
            unique_adj_test_labels = set(all_feats_labels[test_problem][ds_name]['labels'])
            test_fvs_mds, test_labels_mds = decompose_df_feats_labels(
                mds_df, unique_labels = unique_adj_test_labels)
            test_clf = get_svm(test_fvs_mds, test_labels_mds)
            test_labels_svm = test_clf.predict(test_fvs_mds)
            print(test_labels_svm)

            # Get accuracy across all SVM's labels and true labels for a test problem
            test_svm_acc = accuracy_score(test_labels_mds, test_labels_svm)


            # 2. Predict Predictor SVM to test problem
            predictor_labels_svm = predictor_clf.predict(test_fvs_mds)

            # Since Predictor SVM outputs are 0s and 1s, we have to adjust them back to the
            # adjusted test label values (2,3,4,5...)
            class0 = min(unique_adj_test_labels)
            class1 = max(unique_adj_test_labels)
            predictor_labels_svm = [class0 if l == 0 else class1 for l in predictor_labels_svm]

            # Get accuracy across all SVM's labels and true labels for a test problem
            predictor_svm_acc = accuracy_score(test_labels_mds, predictor_labels_svm)

            # Save SVM accuracy to summary output
            if test_problem not in svm_mds_output:
                svm_mds_output[test_problem] = {}
            svm_mds_output[test_problem][ds_name] = {
                'svm': test_clf,
                'm': test_clf.coef_[0],
                'b': test_clf.intercept_[0],
                'test_svm_acc': test_svm_acc,
                'predictor_svm_acc':predictor_svm_acc
            }

    print('Duration: {} seconds'.format(time.time() - start))

    # Return summary output
    return svm_mds_output


def decompose_df_feats_labels(df, unique_labels):
    #
    problem_df = df[df.labels.isin( unique_labels)]

    # Split predictor MDS results into feature vectors and labels
    fvs = problem_df.loc[:, problem_df.columns != 'labels'].to_numpy()
    labels = problem_df['labels'].to_list()
    return fvs, labels


def save_svm_output(svm_output, model_name, predictor_problem, dataset_name, test_problems_name = None, mds=False):
    '''
    '''

    # Create folder for model output, if doesn't exist
    folder = os.path.join('models', model_name, 'svm_output')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename ending
    if test_problems_name is not None:
        ending = dataset_name + '_' + str(test_problems_name)
    else:
        ending = dataset_name

    if mds:
        svm_ver = '_svm_mds_'
    else:
        svm_ver = '_svm_'

    # Construct filename
    fname = str(predictor_problem) + svm_ver + model_name + '_' + ending + '.pickle'
    PATH = os.path.join(folder, fname)

    with open(PATH, 'wb') as f:
        pickle.dump(svm_output, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_svm_output(model_name, predictor_problem, dataset_name, test_problems_name, mds=False):
    '''
    '''

    # Filename ending
    if test_problems_name is not None:
        ending = dataset_name + '_' + str(test_problems_name)
    else:
        ending = dataset_name

    if mds:
        svm_ver = '_svm_mds_'
    else:
        svm_ver = '_svm_'

    # Construct filenames
    fname = str(predictor_problem) + svm_ver + model_name + '_' + ending + '.pickle'
    PATH = os.path.join('models', model_name, 'svm_output', fname)

    # Load features and labels
    with open(PATH, 'rb') as fname:
        svm_output = pickle.load(fname)

    return svm_output


##########################################################################################
# SVM + MDS
##########################################################################################


def get_feats_labels(model_name, problem, dataset_name):
    '''
    '''

    # Construct filenames
    feats_fname = str(problem) + '_test_feats_' + model_name + '_' + dataset_name + '.pickle'
    labels_fname = str(problem) + '_test_labels_' + model_name + '_' + dataset_name + '.pickle'
    PATH_feats = os.path.join('models', model_name, feats_fname)
    PATH_labels = os.path.join('models', model_name, labels_fname)

    # Load features and labels
    with open(PATH_feats, 'rb') as fname:
        feats = pickle.load(fname)
    with open(PATH_labels, 'rb') as fname:
        labels = pickle.load(fname)

    return feats, labels


def get_test_info(model_name, test_problems, dataset_name):
    '''
    '''

    # Make sure test_problems is a list
    test_problems = [test_problems] if type(test_problems) != list else test_problems

    # Keys are test problem numbers and values are lists of features and labels
    test_info = {}

    # Create new key entry in test_info by iterating over all test problems
    for test_problem in test_problems:

        # Get features and labels from files
        test_feats, test_labels = get_feats_labels(
            model_name, test_problem, dataset_name)

        # Add entry to test_info
        test_info[test_problem] = [test_feats, test_labels]
    
    return test_info


def get_all_feats_labels(model_name, predictor_problem, dataset_name, test_problems, test_problems_name, layer_name):
    '''
    '''

    # Get predictor info for predictor problem
    # dataset_name argument is always 'svrt' here since grabbing
    # the trained model's feature embeddings and labels
    predictor_feats, predictor_labels = get_feats_labels(
        model_name, predictor_problem, 'svrt')

    # Get test info for test problems
    test_info = get_test_info(
        model_name, test_problems, dataset_name
    )

    # Collate all predictor and test feats, labels, and unique set of labels,
    # primarily for the reason to eliminate redunancies and eliminate
    # distinctions between predictor and test problems features/labels
    all_feats_labels = {
        predictor_problem: {
            'svrt': {
                'predictor_problem': predictor_problem,
                'feats': predictor_feats, 
                'labels': predictor_labels,
                'unique_labels': [0,1]
            }
        }
    }

    # Iterate append relevant information of test problems to jointly infer MDS
    # with both predictor and test problems
    i_problem = 1
    for test_problem in test_info:

        # Get feats and labels for specific test problem
        test_feats, test_labels = test_info[test_problem]

        # Adjust test labels so that they are not all (0,1), but increase (0,1,2,3...)
        # This is for plotting purposes
        if dataset_name in ['psvrt'] or test_problems_name in ['io', 'sd', 'io2']:
            adj_test_labels = [l + 2*i_problem for l in test_labels]
        else:
            adj_test_labels = test_labels

        # Get new dataset name to distinguish e.g. 'svrt' from 'svrt' on 'io'
        if test_problems_name is not None:
            full_dataset_name = dataset_name + '_' + test_problems_name
        else: 
            full_dataset_name = dataset_name

        # Since appending to dictionary, there will be no duplicates of problems.
        # For instance, if predictor and test are same problem, then there will be
        # only one copy in this dictionary
        if test_problem not in all_feats_labels:
            all_feats_labels[test_problem] = {}

        all_feats_labels[test_problem][full_dataset_name] = {
            'predictor_problem': predictor_problem,
            'feats': test_feats, 
            'labels': adj_test_labels,
            'unique_labels': set(adj_test_labels)
        }

        # Increase problem number for next iteration
        i_problem += 1

    # Create additional storage variables used by MDS based on dictionary after 
    # going through all test problems
    all_feats = {layer_name: []}
    all_labels = []
    for problem in all_feats_labels:
        for ds_name in all_feats_labels[problem]:

            # Get feats and labels for a problem for a particular dataset
            feats = all_feats_labels[problem][ds_name]['feats']
            labels = all_feats_labels[problem][ds_name]['labels']

            all_feats[layer_name] += feats[layer_name]
            all_labels += labels

    return all_feats, all_labels, all_feats_labels


