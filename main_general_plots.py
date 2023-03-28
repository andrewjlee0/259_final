import pickle
import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# with open('models/alexnet/1_loss_alexnet.pickle', 'rb') as fname:
#     x = pickle.load(fname)
#     print(y_acc)

# with open('models/alexnet/1_test_feats_alexnet.pickle', 'rb') as fname:
#     x = pickle.load(fname)
#     print(y_acc)

# with open('models/alexnet/1_test_labels_alexnet.pickle', 'rb') as fname:
#     x = pickle.load(fname)
#     print(y_acc)


# with open('models/resnet18/16_acc_resnet18_svrt.pickle', 'rb') as fname:
#     x = pickle.load(fname)
#     print(x['test'])

# with open('models/resnet18/svm_output/4_svm_resnet18_svrt_io.pickle', 'rb') as fname:
#     x = pickle.load(fname)
#     print('SVM:', x['penult'])

# with open('models/resnet18/23_acc_resnet18_generalize_4.pickle', 'rb') as fname:
#     x = pickle.load(fname)
#     print('SVRT Model 4 Accuracy on Problem 23:', x['test'])



def get_test_acc(model_names, test_problems, dataset_name):
    '''
    '''

    model_names = [model_names] if type(model_names) is not list else model_names

    # Store accuracies, where keys are problems, values are accuracy
    acc_list = []

    # Iterate over all models and test acc files, appending relevant info
    for model_name in model_names:
        for problem in test_problems:

            svrt_fname = str(problem) + '_acc_' + model_name + '_svrt.pickle'
            psvrt_fname = str(problem) + '_acc_' + model_name + '_psvrt.pickle'
            PATH_svrt = os.path.join('models', model_name, svrt_fname)
            PATH_psvrt = os.path.join('models', model_name, psvrt_fname)

            # Get test acc on current SVRT problem and PSVRT
            with open(PATH_svrt, 'rb') as PATH_svrt:
                svrt_acc = pickle.load(PATH_svrt)
            with open(PATH_psvrt, 'rb') as PATH_psvrt:
                psvrt_acc = pickle.load(PATH_psvrt)

            # Add new column for whether problem is SD or SR
            sd_problems = [16, 15, 5, 22, 1, 19, 21, 7, 20]
            if problem in sd_problems:
                test_problems_name = 'Same-Different'
            else:
                test_problems_name = 'Spatial Relation'
                
            # Append svrt acc and psvrt acc
            if not (problem == 1 and model_name == 'alexnet'):
                svrt_acc = svrt_acc['test'][0]
                psvrt_acc = psvrt_acc['test'][0]

            # Construct relevant information for storage variable
            svrt_row = (problem, model_name, svrt_acc, 'svrt', test_problems_name)
            psvrt_row = (problem, model_name, psvrt_acc, 'psvrt', test_problems_name)
            acc_list.append( svrt_row)
            acc_list.append( psvrt_row)

    return acc_list


def get_io_generalize_test_acc():
    '''
    '''

    acc_list = []

    PATH_4p2 = 'models/resnet18/4_acc_resnet18_svrt_2.pickle'
    PATH_4p23 = 'models/resnet18/4_acc_resnet18_svrt_23.pickle'

    PATH_23p2 = 'models/resnet18/23_acc_resnet18_svrt_2.pickle'
    PATH_23p4 = 'models/resnet18/23_acc_resnet18_svrt_4.pickle'

    # PATH_2p4 = 'models/resnet18/2_acc_resnet18_svrt_4.pickle'
    # PATH_2p23 = 'models/resnet18/2_acc_resnet18_svrt_23.pickle'

    # Get test acc on current SVRT problem and PSVRT
    with open(PATH_4p2, 'rb') as PATH_4p2:
        _4p2_acc = round(pickle.load(PATH_4p2)['test'][0] , 2)
    with open(PATH_4p23, 'rb') as PATH_4p23:
        _4p23_acc = round(pickle.load(PATH_4p23)['test'][0] , 2)
    with open(PATH_23p2, 'rb') as PATH_23p2:
        _23p2_acc = round(pickle.load(PATH_23p2)['test'][0] , 2)
    with open(PATH_23p4, 'rb') as PATH_23p4:
        _23p4_acc = round(pickle.load(PATH_23p4)['test'][0] , 2)
    # with open(PATH_2p4, 'rb') as PATH_2p4:
    #     _2p4_acc = round(pickle.load(PATH_2p4)['test'][0] , 2)
    # with open(PATH_2p23, 'rb') as PATH_2p23:
    #     _2p23_acc = round(pickle.load(PATH_2p23)['test'][0] , 2)

    # # 
    # PATH_4svm = 'models/resnet18/svm_output/4_svm_resnet18_svrt_io.pickle'
    # PATH_23svm = 'models/resnet18/svm_output/23_svm_resnet18_svrt_io2.pickle'
    # PATH_4svm_mds = 'models/resnet18/svm_output/4_svm_mds_resnet18_svrt_io.pickle'
    # PATH_23svm_mds = 'models/resnet18/svm_output/23_svm_mds_resnet18_svrt_io2.pickle'
    # for path in [PATH_4svm, PATH_23svm, PATH_4svm_mds, PATH_23svm_mds]:
    #     with open(path, 'rb') as path:
    #         x = pickle.load(path)
    #         print(x)

    PATH_16svm = 'models/resnet18/svm_output/16_svm_resnet18_svrt_sd.pickle'
    PATH_16svm_mds = 'models/resnet18/svm_output/16_svm_mds_resnet18_svrt_sd.pickle'
    for path in [PATH_16svm, PATH_16svm_mds]:
        with open(path, 'rb') as path:
            x = pickle.load(path)

            for test_problem in x:
                if type(test_problem) is not int:
                    continue

                for dsname in x[test_problem]:
                    print(path, '####################################################')
                    print('test_svm_acc', test_problem, x[test_problem][dsname]['test_svm_acc'])
                    print('predictor_svm_acc', test_problem, x[test_problem][dsname]['predictor_svm_acc'])


    # with open(PATH_4svm, 'rb') as PATH_4svm:
    #     _4svm = pickle.load(PATH_4svm)
    #     _4svm2_test_acc = round(_4svm[2]['svrt']['test_svm_acc'] , 2)
    #     _4svm23_acc = round(_4svm[23]['svrt']['acc'] , 2)


    # with open(PATH_23svm, 'rb') as PATH_23svm:
    #     _23svm = pickle.load(PATH_23svm)['penult']['test']
    #     _23svm2_acc = round(_23svm[2]['svrt_io2']['acc'] , 2)
    #     _23svm4_acc = round(_23svm[4]['svrt_io2']['acc'] , 2)

    # # Construct relevant information for storage variable
    # _4p2_row = (4, 'resnet18', _4p2_acc, 'svrt', 2, _4svm2_acc)
    # _4p23_row = (4, 'resnet18', _4p23_acc, 'svrt', 23, _4svm23_acc)
    # _23p2_row = (23, 'resnet18', _23p2_acc, 'svrt', 2, _23svm2_acc)
    # _23p4_row = (23, 'resnet18', _23p4_acc, 'svrt', 4, _23svm4_acc)
    # acc_list.append( _4p2_row)
    # acc_list.append( _4p23_row)
    # acc_list.append( _23p2_row)
    # acc_list.append( _23p4_row)

    # #
    # PATH_16svm = 'models/resnet18/svm_output/16_svm_resnet18_svrt_sd.pickle'

    # with open(PATH_16svm, 'rb') as PATH_16svm:
    #     _16svm = pickle.load(PATH_16svm)['penult']['test']

    # for test_problem in [15, 5, 22, 1, 19, 21, 7, 20]:
    #     PATH_16pn = 'models/resnet18/16_acc_resnet18_svrt_' + str(test_problem) + '.pickle'

    #     with open(PATH_16pn, 'rb') as PATH_16pn:
    #         _16pn_acc = round(pickle.load(PATH_16pn)['test'][0] , 2)
    #         _16svmn_acc = round(_16svm[test_problem]['svrt_sd']['acc'] , 2)

    #         _16pn_row = (16, 'resnet18', _16pn_acc, 'svrt', test_problem, _16svmn_acc)
    #         acc_list.append( _16pn_row)

    return acc_list


def plot_svrt_test_acc(acc_list):
    '''
    '''

    cols = ['problem', 'model_name', 'test_acc', 'dataset_name', 'test_problems_name']
    plot_df = pd.DataFrame(data = acc_list, columns = cols)
    plot_df['model_name'] = plot_df['model_name'].map({
        'alexnet': 'AlexNet',
        'resnet18': 'ResNet-18'
    })
    plot_df = plot_df[plot_df.dataset_name == 'svrt']

    sns.set_context('notebook')
    fig = plt.figure(figsize = (12,3))
    fig.tight_layout()
    print(plot_df[plot_df.model_name == 'ResNet-18'].sort_values(['test_acc']))

    sns.barplot(
        data = plot_df,
        x = 'problem',
        y = 'test_acc',
        hue = 'model_name',
        order = plot_df[plot_df.model_name == 'ResNet-18'].sort_values(['test_acc'])['problem']
    ).set(
        ylabel = 'Test Accuracy on SVRT Problem',
        xlabel = 'SVRT Problem',
        ylim = [0.4, 1],
    )
    plt.legend(frameon=False, loc='upper left')
    sns.despine()

    plt.axvline(7.5, color='red', label='Same-Different')

    fname = 'test_acc_svrt.png'
    PATH = os.path.join('general_plots', fname)
    plt.savefig(PATH, bbox_inches = 'tight')


def plot_psvrt_test_acc(acc_list):
    '''
    '''

    cols = ['problem', 'model_name', 'test_acc', 'dataset_name', 'test_problems_name']
    plot_df = pd.DataFrame(data = acc_list, columns = cols)
    plot_df['dataset_name'] = plot_df['dataset_name'].map({
        'svrt': 'SVRT',
        'psvrt': 'PSVRT'
    })
    plot_df = plot_df[(plot_df.model_name == 'resnet18') & (plot_df.dataset_name == 'PSVRT')]
     
    mean_sd = plot_df[plot_df.test_problems_name == 'Same-Different']['test_acc'].mean()
    mean_sr = plot_df[plot_df.test_problems_name == 'Spatial Relation']['test_acc'].mean()

    std_sd = plot_df[plot_df.test_problems_name == 'Same-Different']['test_acc'].std()
    std_sr = plot_df[plot_df.test_problems_name == 'Spatial Relation']['test_acc'].std()

    print('SD:', mean_sd, std_sd)
    print('SR:', mean_sr, std_sr)

    sns.set_context('notebook')
    fig = plt.figure()
    fig.tight_layout()

    sns.barplot(
        data = plot_df,
        x = 'test_problems_name',
        y = 'test_acc',
    ).set(
        ylabel = 'Accuracy on Same-Different PSVRT',
        xlabel = 'SVRT Problem Type',
        ylim = [0.4, 0.6],
    )
    plt.legend(frameon=False, loc='upper left')
    sns.despine()

    plt.axhline(0.5, color='red')


    # fig = plt.figure(figsize = (12,3))
    # fig.tight_layout()

    # sns.barplot(
    #     data = plot_df,
    #     x = 'problem',
    #     y = 'test_acc',
    #     hue = 'dataset_name',
    #     order = plot_df[plot_df.dataset_name == 'SVRT'].sort_values(['test_acc'])['problem']
    # ).set(
    #     ylabel = 'Test Accuracy on PSVRT and SVRT Problems',
    #     xlabel = 'SVRT Problem',
    #     ylim = [0.4, 1],
    # )
    # plt.legend(frameon=False, loc='upper left')
    # sns.despine()

    # plt.axvline(7.5, color='red', label='Same-Different')

    fname = 'test_acc_psvrt.png'
    PATH = os.path.join('general_plots', fname)
    plt.savefig(PATH, bbox_inches = 'tight')


def table_io_test_acc(acc_list):
    '''
    '''

    cols = ['Trained On', 'model_name', 'Accuracy', 'dataset_name', 'Generalize To', 'MDS + SVM Accuracy']
    plot_df = pd.DataFrame(data = acc_list, columns = cols)

    plot_df.drop(['dataset_name', 'model_name'], axis=1)

    plot_df = plot_df[['Trained On', 'Generalize To', 'Accuracy', 'MDS + SVM Accuracy']]
    plot_df = plot_df.to_latex(
        # column_format="rrrr", position="h", 
        # label="table:5", 
    )


    # sns.set_context('notebook')
    # fig = plt.figure()
    # fig.tight_layout()

    # sns.barplot(
    #     data = plot_df,
    #     x = 'problem',
    #     y = 'test_acc'
    # ).set(
    #     ylabel = 'Accuracy on Other Inside-Outside Problems',
    #     xlabel = 'SVRT Problem Type',
    #     ylim = [0.4, 0.6],
    # )
    # plt.legend(frameon=False, loc='upper left')
    # sns.despine()

    # plt.axhline(0.5, color='red')

    fname = 'test_acc_io.png'
    PATH = os.path.join('general_plots', fname)
    plt.savefig(PATH, bbox_inches = 'tight')


# acc_list = get_test_acc(['alexnet', 'resnet18'], list(range(1, 24)), 'svrt')
# plot_svrt_test_acc(acc_list)
# plot_psvrt_test_acc(acc_list)

acc_list = get_io_generalize_test_acc()
table_io_test_acc(acc_list)



