"""
functions:
run_machine_learning()
calc_accuracy()

get_haars_df(rows=5)
get_mf_df(rows=5)
get_mf_pca_df(rows=5)
get_time_df()
"""

from constants import *
import pandas as pd
import os
from data_loader import load_data, load_dataframe
from ml_train_predict import training, prediction
from sklearn.metrics import accuracy_score
from data_saver import save_dataframe


def run_machine_learning():
    """
    function run_machine_learning() creates all dataframes,
    calls function for loading data and starts the process
    of machine learning
    """
    dataframes_names = ['haars_df', 'matrix_df', 'matrix_pca_df']
    dataframes = [pd.DataFrame(columns=COL_LIST) for i in range(3)]
    types = ['hf', 'mf', 'mf_pca']
    all_features, y_train, y_test, time_df = load_data()

    # train model for all feature type and save  predicted results
    for i in range(3):
        training(all_features[i], y_train, types[i], time_df, i)
        prediction(all_features[i+3],  types[i], dataframes[i], time_df, i+3)
        dataframes[i].insert(0, 'y_test', y_test)
        # add column for seeing the result of voting
        # 4 methods using their predicted results
        dataframes[i]['voting (cols [1-5) )'] = \
            round(dataframes[i][dataframes[i].columns[1:5]].mean(axis=1))

        save_dataframe(dataframes[i], dataframes_names[i])
    save_dataframe(time_df, 'time_df_part1')


def calc_accuracy():
    """
    function calc_accuracy() calculate accuracy
    and returns dataframe with results
    """
    col_list = COL_LIST
    col_list.append('voting (cols [1-5) )')

    haars_df = load_dataframe('haars_df')
    matrix_df = load_dataframe('matrix_df')
    matrix_pca_df = load_dataframe('matrix_pca_df')

    accuracy_df = pd.DataFrame(FEATURES_LIST, index=['1', '2', '3'], columns=['features'])

    for i, cols in enumerate(col_list):
        accuracy_df[cols] = [accuracy_score(haars_df['y_test'], haars_df[cols]),
                             accuracy_score(matrix_df['y_test'], matrix_df[cols]),
                             accuracy_score(matrix_pca_df['y_test'], matrix_pca_df[cols])]
    return accuracy_df[:]


def get_haars_df(rows=5):
    """
    function get_haars_df(rows=5) shows some results
    that were predicted for haars features by all methods
    """
    haars_df = load_dataframe('haars_df')
    return haars_df[:rows]


def get_mf_df(rows=5):
    """
    function get_mf_df(rows=5) shows some results
    that were predicted for matrix features by all methods
    """
    mf_df = load_dataframe('matrix_df')
    return mf_df[:rows]


def get_mf_pca_df(rows=5):
    """
    function get_mf_pca_df(rows=5) shows some results
    that were predicted for matrix+pca features by all methods
    """
    mf_pca_df = load_dataframe('matrix_pca_df')
    return mf_pca_df[:rows]


def get_time_df():
    """
    function get_time_df() returns dataframe with time
    """
    time_part1_df = load_dataframe('time_df_part1')
    return time_part1_df[:]


if __name__ == '__main__':
    run_ml()
