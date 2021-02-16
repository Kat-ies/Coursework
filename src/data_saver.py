"""
functions:
save_dataframe(dataframe, filename, folder='Dataframes', path=PATH)
save_data(all_features, x_train, x_test, y_test, y_train, path=PATH)
save_ml_model(model, name, path=PATH, folder='Classificators')
"""

import os
import joblib
import pandas as pd
from constants import *


def save_dataframe(dataframe, filename, folder='Dataframes', path=PATH):
    """
    function save_dataframe(dataframe, filename, folder='Dataframes', path=PATH)
    saves input dataframe
    """
    dataframe.to_csv(os.path.join(path, folder, filename + '.csv'))


def save_data(all_features, x_train, x_test, y_test, y_train, path=PATH):
    """
    function save_data(all_features, x_train, x_test, y_test, y_train, path=PATH)
    saves all created data to the given path of to the default path
    """
    joblib.dump(all_features, os.path.join(path, 'all_features.pkl'))
    joblib.dump(y_train, os.path.join(path, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(path, 'y_test.pkl'))
    joblib.dump(x_train, os.path.join(path, 'x_train.pkl'))
    joblib.dump(x_test, os.path.join(path, 'x_test.pkl'))


def save_ml_model(model, name, path=PATH, folder='Classificators'):
    """
    function save_ml_model(model, name, path=PATH, folder='Classificators')
    saves model trained by ml-method
    """
    joblib.dump(model, os.path.join(path, folder, name + '.pkl'))
