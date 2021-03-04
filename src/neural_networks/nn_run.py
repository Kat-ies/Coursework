"""
functions:
run_fully_connected_network(experiment_number=1)
run_convolution_network()
calc_accuracy(exp_num=1)
get_haars_net_df(rows=5, exp_num=1)
get_matrix_net_df(rows=5, exp_num=1)
get_matrix_net_pca_df(rows=5, exp_num=1)
get_net_time_df(exp_num=1)
"""

from constants import *
import pandas as pd
from nn_train_predict import *
from data_loader import load_data, load_nn_dataframes, load_dataframe
from data_saver import save_nn_dataframes
from sklearn.metrics import accuracy_score


def run_fully_connected_network(experiment_number=1):
    """
    function run_fully_connected_network(experiment_number=1)
    creates all dataframes, calls function for loading data
    and runs training and prediction functions for
    fully connected neural network with params depending on experiment
    """
    all_features, y_train, y_test, time_df = load_data()
    time_of_features_creat = time_df[COL_LIST[0]].values.tolist()

    net_time_df = pd.DataFrame(index=CATEGORIES, columns=NET_COLS)

    # fist add time for features creation
    for i, rows in enumerate(CATEGORIES):
        if rows[:2] == 'mf':
            net_time_df.loc[rows] = time_of_features_creat[i]
        else:
            net_time_df.loc[rows][0] = net_time_df.loc[rows][1] = time_of_features_creat[i]

    haars_net_df = pd.DataFrame(columns=NET_COLS[:2])
    matrix_net_df = pd.DataFrame(columns=NET_COLS)
    matrix_net_pca_df = pd.DataFrame(columns=NET_COLS[:2])
    y_points_acc, y_points_loss, y_points_val = [], [], []

    range_num = 0
    lr = 0.001
    if experiment_number == 2:
        range_num = 1
        lr = 0.0001

    for is_gpu in range(range_num, 2):
        nn_training(all_features[0], y_train, is_gpu, net_time_df, 'hf', 0, y_points_acc, y_points_loss,
                    y_points_val, 'fully_connected', learning_rate=lr)
        nn_test(all_features[3], is_gpu, 'hf', haars_net_df, net_time_df, 3, 'fully_connected')

        nn_training(all_features[1], y_train, is_gpu, net_time_df, 'mf', 1, y_points_acc, y_points_loss,
                    y_points_val, 'fully_connected', learning_rate=lr)
        nn_test(all_features[4], is_gpu, 'mf', matrix_net_df, net_time_df, 4, 'fully_connected')

        nn_training(all_features[2], y_train, is_gpu, net_time_df, 'mf_pca', 2, y_points_acc, y_points_loss,
                    y_points_val, 'fully_connected', learning_rate=lr)
        nn_test(all_features[5], is_gpu, 'mf_pca', matrix_net_pca_df, net_time_df, 5, 'fully_connected')

    haars_net_df.insert(0, 'y_test', y_test)
    matrix_net_df.insert(0, 'y_test', y_test)
    matrix_net_pca_df.insert(0, 'y_test', y_test)

    save_nn_dataframes(haars_net_df, matrix_net_df, matrix_net_pca_df, net_time_df,
                       experiment_number)


def run_convolution_network():
    """
    function  run_convolution_network()
    creates all dataframes, calls function for loading data
    and runs training and prediction functions for
    convolution neural network
    """
    haars_net_df, matrix_net_df, matrix_net_pca_df, net_time_df = load_nn_dataframes()
    all_features, y_train, y_test, time_df = load_data()

    y_points_acc, y_points_loss, y_points_val = [], [], []
    for is_gpu in range(0, 2):
        y_points_acc.clear(), y_points_loss.clear(), y_points_val.clear()
        nn_training(all_features[1], y_train, is_gpu, net_time_df, 'mf',
                    1, y_points_acc, y_points_loss,
                    y_points_val, 'convolution', loop=MAX_ITER // 50)
        nn_test(all_features[4], is_gpu, 'mf', matrix_net_df, net_time_df, 4, 'convolution')

    save_nn_dataframes(haars_net_df, matrix_net_df, matrix_net_pca_df, net_time_df)


def calc_accuracy(exp_num=1):
    """
    function calc_accuracy() calculate accuracy
    and returns dataframe with results
    """
    haars_net_df, matrix_net_df, matrix_net_pca_df, net_time_df = load_nn_dataframes(exp_num)

    net_accuracy = pd.DataFrame(FEATURES_LIST, index=['1', '2', '3'], columns=['features'])
    net_accuracy[NET_COLS[1]] = [accuracy_score(haars_net_df['y_test'], haars_net_df[NET_COLS[1]]),
                                 accuracy_score(matrix_net_df['y_test'], matrix_net_df[NET_COLS[1]]),
                                 accuracy_score(matrix_net_pca_df['y_test'], matrix_net_pca_df[NET_COLS[1]])]

    if exp_num == 1:
        net_accuracy[NET_COLS[3]] = [np.NaN, accuracy_score \
            (matrix_net_df['y_test'], matrix_net_df[NET_COLS[3]]), np.NaN]

    return net_accuracy[:]


def get_haars_net_df(rows=5, exp_num=1):
    """
    function get_haars_net_df(rows=5, exp_num=1) shows some results
    that were predicted for haars features by neural networks at
    first or second experiment
    """
    haars_net_df = load_dataframe('haars_net_df' + '_exp_num_' + str(exp_num))
    return haars_net_df[:rows]


def get_matrix_net_df(rows=5, exp_num=1):
    """
    function get_matrix_net_df(rows=5, exp_num=1) shows some results
    that were predicted for haars features by neural networks at
    first or second experiment
    """
    matrix_net_df = load_dataframe('matrix_net_df' + '_exp_num_' + str(exp_num))
    return matrix_net_df[:rows]


def get_matrix_net_pca_df(rows=5, exp_num=1):
    """
    function get_matrix_net_pca_df(rows=5, exp_num=1) shows some results
    that were predicted for haars features by neural networks at
    first or second experiment
    """
    matrix_net_pca_df = load_dataframe('matrix_net_pca_df' + '_exp_num_' + str(exp_num))
    return matrix_net_pca_df[:rows]


def get_net_time_df(exp_num=1):
    """
    function get_net_time_df(exp_num=1) returns dataframe with time
    """
    net_time_df = load_dataframe('net_time_df' + '_exp_num_' + str(exp_num))
    return net_time_df[:]
