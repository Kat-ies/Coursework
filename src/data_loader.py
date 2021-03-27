""""
functions:
unpacking_zips()
load_dataframe(df_name, path=PROJECT_PATH, folder_name='Dataframes')
load_sample(sample_name, path=PROJECT_PATH)
load_data(path=PROJECT_PATH)
load_ml_model(name, path=PROJECT_PROJECT_PATH, folder='Classificators')
def load_nn_model(name, path=PROJECT_PATH, folder='Networks')
"""

import os
import joblib
import zipfile
import torch
from constants import *
from pandas import read_csv
import requests
import torchvision
from data_loader import download_file_from_google_drive
from constants import *
from face_detection.retrain_model import set_device, get_object_detection_model


def unpacking_zips():
    """
    function unpacking_zips() unpacks zip files from the list of directories
    now we need only one archive, when we need more, we'll add new paths
    to the list and easily unpack all archives at once"""
    for zips in DIRECTORIES:
        (zipfile.ZipFile(zips, 'r')).extractall()


def load_dataframe(df_name, path=PROJECT_PATH, folder_name='Dataframes'):
    """
    function load_dataframe(df_name, path=PROJECT_PATH, folder_name='Dataframes')
    loads dataframe with df_name
    """
    return read_csv(os.path.join(path, folder_name,
                                 df_name + '.csv'), index_col=0, header=0)


def load_sample(sample_name, path=PROJECT_PATH):
    """
    function load_sample(sample_name, path=PROJECT_PATH)
    load sample with sample_name name
    """
    sample = joblib.load(os.path.join(path, sample_name + '.pkl'))
    return sample


def load_data(path=PROJECT_PATH):
    """
    function load_features(path=PROJECT_PATH)
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


def load_ml_model(name, path=PROJECT_PATH, folder='Classificators'):
    """
    function load_ml_model(name, path=PROJECT_PATH, folder='Classificators')
    loads trained models for the machine learning part
    """
    return joblib.load(os.path.join(path, folder, name + '.pkl'))


def load_nn_model(name, path=PROJECT_PATH, folder='Networks'):
    """
    function  load_nn_model(name, path=PROJECT_PATH, folder='Networks')
    loads trained models for the neural networks part
    """
    return torch.load(os.path.join(path, folder, name + '.pth'))


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def load_model(mode='USER', trained=False, model_name='faster_rcnn', id=FASTER_RCNN_ID):
    # model_name is necessary for adding more nets (in future)
    try:
        device = set_device()
        model = get_object_detection_model()
        model.to(device)

        if mode == 'USER':

            download_file_from_google_drive(id, os.path.join(MODEL_PATH, model_name + '.pth'))
            model.load_state_dict(os.path.join(MODEL_PATH, model_name + '.pth'))
            model.eval()

        elif mode == 'ADMIN':
            if trained:
                model.load_state_dict(load_nn_model('faster_rcnn', path=WORK_PATH, folder='Models'))
                model.eval()
        else:
            raise RuntimeError('Invalid working mode')

        return model

    except RuntimeError as re:
        print(*re.args)


if __name__ == '__main__':
    unpacking_zips()
