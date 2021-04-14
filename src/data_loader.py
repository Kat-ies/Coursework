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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from constants import *
from torchvision.models.detection.retinanet import RetinaNetHead


# from face_detection.retrain_model import get_object_detection_model
# строка 20 у меня почему-то не работает:( Только если так:


def get_object_detection_model(pretrained=True, model_name='faster_rcnn'):
    try:
        # replace the classifier with a new one, that has num_classes which is user-defined
        num_classes = 2  # 1 + background

        if model_name == 'faster_rcnn' or model_name == 'faster_rcnn_not_pretrained':
            # load an object detection model pre-trained on COCO == pretrained=True
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        elif model_name == 'retina_net' or model_name == 'retina_net_not_pretrained':
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained, num_classes=2)

            # get number of input features and anchor boxed for the classifier
            #in_features = model.head.classification_head.conv[0].in_channels
            #num_anchors = model.head.classification_head.num_anchors

            # replace the pre-trained head with a new one
            #model.head = RetinaNetHead(in_features, num_anchors, num_classes)

        else:
            raise RuntimeError('Invalid model name!')

    except RuntimeError as re:
        print(*re.args)

    return model


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


def load_model(mode='USER', trained=False, pretrained=True, model_name='faster_rcnn', id=FASTER_RCNN_ID):
    # model_name is necessary for adding more nets (in future)
    try:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = get_object_detection_model(pretrained, model_name)
        model.to(device)

        '''if mode == 'USER':

            download_file_from_google_drive(id, os.path.join(MODEL_PATH, model_name + '.pth'))
            model.load_state_dict(os.path.join(MODEL_PATH, model_name + '.pth'))
            model.eval()'''

        # elif mode == 'ADMIN':
        if mode == 'ADMIN':
            if trained:
                model.load_state_dict(load_nn_model(model_name, path=WORK_PATH, folder='Models'))
                model.eval()
        else:
            raise RuntimeError('Invalid working mode')

        return model

    except RuntimeError as re:
        print(*re.args)


if __name__ == '__main__':
    unpacking_zips()
