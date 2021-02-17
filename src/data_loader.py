""""
functions:
unpacking_zips()
load_dataframe(df_name, path=PATH, folder_name='Dataframes')
load_sample(sample_name, path=PATH)
load_data(path=PATH)
load_ml_model(name, path=PATH, folder='Classificators')
def load_nn_model(name, path=PATH, folder='Networks')
"""

import os
import joblib
import zipfile
import torch
from constants import *
from pandas import read_csv


def unpacking_zips():
    """
    function unpacking_zips() unpacks zip files from the list of directories
    now we need only one archive, when we need more, we'll add new paths
    to the list and easily unpack all archives at once"""
    for zips in DIRECTORIES:
        (zipfile.ZipFile(zips, 'r')).extractall()


def load_dataframe(df_name, path=PATH, folder_name='Dataframes'):
    """
    function load_dataframe(df_name, path=PATH, folder_name='Dataframes')
    loads dataframe with df_name
    """
    return read_csv(os.path.join(path, folder_name,
                                 df_name + '.csv'), index_col=0, header=0)


def load_sample(sample_name, path=PATH):
    """
    function load_sample(sample_name, path=PATH)
    load sample with sample_name name
    """
    sample = joblib.load(os.path.join(path, sample_name + '.pkl'))
    return sample


def load_data(path=PATH):
    """
    function load_features(path=PATH)
    loads all necessary data for experiments
    """
    all_features, y_train, y_test = load_sample('all_features'), \
                                    load_sample('y_train'), \
                                    load_sample('y_test')
    time_df = load_dataframe('time_df_part0')

    return all_features, y_train, y_test, time_df


def load_nn_dataframes(exp_num=1):
    return load_dataframe('haars_net_df' + '_exp_num_' + str(exp_num)), \
           load_dataframe('matrix_net_df' + '_exp_num_' + str(exp_num)), \
           load_dataframe('matrix_net_pca_df' + '_exp_num_' + str(exp_num)), \
           load_dataframe('net_time_df' + '_exp_num_' + str(exp_num))


def load_ml_model(name, path=PATH, folder='Classificators'):
    """
    function load_ml_model(name, path=PATH, folder='Classificators')
    loads trained models for the machine learning part
    """
    return joblib.load(os.path.join(path, folder, name + '.pkl'))


def load_nn_model(name, path=PATH, folder='Networks'):
    """
    function  load_nn_model(name, path=PATH, folder='Networks')
    loads trained models for the neural networks part
    """
    return torch.load(os.path.join(path, folder, name + '.pth'))


if __name__ == '__main__':
    unpacking_zip()
