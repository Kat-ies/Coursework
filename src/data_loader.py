""""
functions:
unpacking_zips()
"""

import os
import joblib
import zipfile
from constants import *


def unpacking_zips():
    """
    function unpacking_zips() unpacks zip files from the list of directories
    now we need only one archive, when we need more, we'll add new paths
    to the list and easily unpack all archives at once"""
    for zips in DIRECTORIES:
        (zipfile.ZipFile(zips, 'r')).extractall()


def ml_method(path):
    x_test = joblib.load(os.path.join(path, 'x_test.pkl'))
    time_df = joblib.load(os.path.join(path, 'Dataframes', 'time_df_part0.pkl'))
    return x_test, time_df


def nn_method(path):
    time_of_features_creat = joblib.load(os.path.join(path, 'time_of_features_creat.pkl'))
    return time_of_features_creat


def load_features(method_type, path=PATH):
    all_features = joblib.load(os.path.join(path, 'all_features.pkl'))
    y_train = joblib.load(os.path.join(path, 'y_train.pkl'))
    y_test = joblib.load(os.path.join(path, 'y_test.pkl'))
    if method_type == 'ml_meth':
        x_test, time_df = ml_method(path)
        return all_features, y_train, y_test,  x_test, time_df
    else:
        time_of_features_creat = nn_method(path)
        return all_features, y_train, y_test, time_of_features_creat

# заметки, чтобы потом спросить и сделать код выше красивее
# изначально я хотела сделать такую штуку:
# def meth1():
#     pass
#
# def meth2():
#     pass
#
# def run_meth(method):
#     return method()
#
# run_meth(meth1)
# run_meth(meth2)
# но у меня не получилось запустить такое из ноутбука :(


if __name__ == '__main__':
    unpacking_zip()
