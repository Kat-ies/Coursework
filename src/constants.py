"""
file with all useful constants

Faces, Rectangle - used in • dataloader.py for correct data loading
                           • new_dataset_creation.py for making new dataset
                           • visualization.py for showing images from new dataset

HaarRect - used in • features.py for storing rectangles in haars features

PATH - used in  • dataloader.py for loading features
                • features.py for loading .xml file with
                haars features and saving data
DIRECTORIES - used in  • dataloader.py as a path with archive data

COL_LIST, CATEGORIES -used in  • features.py for creating time dataframe

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
"""
from collections import namedtuple

Faces = namedtuple('Faces', 'img is_face filename rect')
Rectangle = namedtuple('Rectangle', 'x y w h')
HaarRect = namedtuple('HaarRect', 'x y w h weight')

PATH = '/content/drive/My Drive/КУ Курсачи/Курсовой проект 2020/files/'

DIRECTORIES = ['/content/drive/My Drive/КУ Курсачи/'
               'Курсовой проект 2020/WIDER_FACE (zip)/WIDER_train.zip']

COL_LIST = ['logreg_', 'tree_', 'knn_', 'svm_', 'randforest_', 'ada_boost_', 'grad_boost_']
CATEGORIES = ['hf_train', 'mf_train', 'pca_train', 'hf_test', 'mf_test', 'pca_test']

MAXSIZE = (25, 25)
MINSIZE = (20, 20)
RANDOM_BOX = (30, 30, 30, 30)

RANDOM_SEED = 42

COLUMNS_FOR_WF_DATASET = 1
ROWS_FOR_WF_DATASET = 5

COLUMNS_FOR_NEW_DATASET = 5
ROWS_FOR_NEW_DATASET = 5
