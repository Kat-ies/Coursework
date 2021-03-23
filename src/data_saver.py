"""
functions:
save_dataframe(dataframe, filename, folder='Dataframes', path=PROJECT_PATH)
save_data(all_features, x_train, x_test, y_test, y_train, path=PROJECT_PATH)
save_ml_model(model, name, path=PROJECT_PATH, folder='Classificators')

"""

import os
import joblib
import torch
import pandas as pd
from constants import *


def save_dataframe(dataframe, filename, folder='Dataframes', path=PROJECT_PATH):
    """
    function save_dataframe(dataframe, filename, folder='Dataframes', path=PROJECT_PATH)
    saves input dataframe
    """
    dataframe.to_csv(os.path.join(path, folder, filename + '.csv'))


def save_data(all_features, x_train, x_test, y_test, y_train, path=PROJECT_PATH):
    """
    function save_data(all_features, x_train, x_test, y_test, y_train, path=PROJECT_PATH)
    saves all created data to the given path of to the default path
    """
    joblib.dump(all_features, os.path.join(path, 'all_features.pkl'))
    joblib.dump(y_train, os.path.join(path, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(path, 'y_test.pkl'))
    joblib.dump(x_train, os.path.join(path, 'x_train.pkl'))
    joblib.dump(x_test, os.path.join(path, 'x_test.pkl'))


def save_ml_model(model, name, path=PROJECT_PATH, folder='Classificators'):
    """
    function save_ml_model(model, name, path=PROJECT_PATH, folder='Classificators')
    saves model trained by ml-method
    """
    joblib.dump(model, os.path.join(path, folder, name + '.pkl'))


def save_nn_model(param, name, path=PROJECT_PATH, folder='Networks'):
    torch.save(param, os.path.join(path, folder, name + '.pth'))


def save_nn_dataframes(haars_net_df, matrix_net_df, matrix_net_pca_df, net_time_df, exp_num=1):
    save_dataframe(haars_net_df, 'haars_net_df' + '_exp_num_' + str(exp_num))
    save_dataframe(matrix_net_df, 'matrix_net_df' + '_exp_num_' + str(exp_num))
    save_dataframe(matrix_net_pca_df, 'matrix_net_pca_df' + '_exp_num_' + str(exp_num))
    save_dataframe(net_time_df, 'net_time_df' + '_exp_num_' + str(exp_num))
