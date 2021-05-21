"""
file with all useful constants

Faces, Rectangle - used in • dataloader.py for correct data loading
                           • new_dataset_creation.py for making new dataset
                           • visualization.py for showing images from new dataset
                           • images_and_frames.py to make frames

HaarRect - used in • features.py for storing rectangles in haars features

PROJECT_PATH - used in  • data_loader.py for loading features
                • features.py for loading .xml file with
                haars features and saving data
                •images_and_frames.py for loading marked up file with frames
                • data_saver.py for saving files

IMG_PATH - used in  •images_and_frames.py as a root for loading
                    unpacked wider face dataset

DIRECTORIES - used in  • dataloader.py as a path with archive data

FEATURES_LIST - used in • ml_run for creating accuracy dataframe

COL_LIST, CATEGORIES - used in  • features.py for working with dataframes
                                • ml_train_predict for working with dataframes
                                • nn_train_predict for working with dataframes

NET_COLS - used in • nn_train_predict for working with dataframes

MAXSIZE - used in  • features.py for checking image size
                   • new_dataset_creation.py for resizing images

MINSIZE - used in  • new_dataset_creation.py for ignoring too small images

RANDOM_BOX - used in  • new_dataset_creation.py to crop an image,
                      when there is no faces in the image

RANDOM_SEED - used in  • features.py to fix random param for a shuffle
                       • visualization.py to fix random param for
                       a random numbers generator

COLUMNS_FOR_WF_DATASET, ROWS_FOR_WF_DATASET  - used in
                        • visualization.py for showing fix number of
                        wider faces images in each row and column

COLUMNS_FOR_NEW_DATASET, ROWS_NEW_WF_DATASET  - used in
                        • visualization.py for showing fix number of
                        new images in each row and column

COLUMNS_FOR_WRONG_CLF_OBJ, ROWS_FOR_WRONG_CLF_OBJ - used in
                        • visualization.py for showing fix number of
                        new images in each row and column
"""

from collections import namedtuple

Faces = namedtuple('Faces', 'img is_face filename rect')
Rectangle = namedtuple('Rectangle', 'x y w h')
HaarRect = namedtuple('HaarRect', 'x y w h weight')

# for google-drive
PROJECT_PATH = '/content/drive/My Drive/КУ Курсачи/Курсовой проект 2020/files/'
WORK_PATH = '/content/drive/MyDrive/КУ Курсачи/Курсовая работа 2021/'

DIRECTORIES = ['/content/drive/My Drive/КУ Курсачи/'
               'Курсовой проект 2020/WIDER_FACE (zip)/WIDER_train.zip']

IMG_PATH = 'WIDER_train/images'

"""
# for local system
PROJECT_PATH = '/home/katerinka/КУ Курсачи/Курсовой проект 2020/files/'
WORK_PATH = '/home/katerinka/КУ Курсачи/Курсовая работа 2021/'

DIRECTORIES = ['/home/katerinka/КУ Курсачи/'
               'Курсовой проект 2020/WIDER_FACE (zip)/WIDER_train.zip']
               
IMG_PATH = '/home/katerinka/КУ Курсачи/Курсовой проект 2020/WIDER_FACE/WIDER_train/images/'
"""

FEATURES_LIST = ['Haars features', 'Matrix features', 'Matrix + PCA features']
COL_LIST = ['logreg_', 'tree_', 'knn_', 'svm_', 'randforest_', 'ada_boost_', 'grad_boost_']
CATEGORIES = ['hf_train', 'mf_train', 'pca_train', 'hf_test', 'mf_test', 'pca_test']
NET_COLS = ['fully_con_net_cpu', 'fully_con_net_gpu', 'convol_net_cpu', 'convol_net_gpu']

MAXSIZE = (25, 25)
MINSIZE = (20, 20)
RANDOM_BOX = (30, 30, 30, 30)

RANDOM_SEED = 42

COLUMNS_FOR_WF_DATASET = 1
ROWS_FOR_WF_DATASET = 5

COLUMNS_FOR_NEW_DATASET = 5
ROWS_FOR_NEW_DATASET = 5

COLUMNS_FOR_WRONG_CLF_OBJ = 4
ROWS_FOR_WRONG_CLF_OBJ = 24

MAX_ITER = 50000
LEARNING_RATE = 0.001

FASTER_RCNN_ID = '1yGreLksQtlbt0We-9nhgun1yehtL5kJa'
MODEL_PATH = 'face_detection'

SCORE_BACKGROUND_LABEL_WIDTH = 65
SCORE_BACKGROUND_LABEL_HEIGHT = 14

FASTER_LOG_LOSSES = ['loss', 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
RETINA_LOG_LOSSES = ['loss', 'classification', 'bbox_regression']
