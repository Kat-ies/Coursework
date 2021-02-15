from lxml import etree
import sklearn.model_selection as sk
from collections import namedtuple
import pandas as pd
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random

MAXSIZE = (25, 25)

# class HaarFeature is created for calculating the value of every feature
# each haars feature is represented by 2-3 rectangles
# one rectangle looks like <_> x1, y1, w, h, weight </_>
# (top left corner + width + height + weight)
# for more details see https://api-2d3d-cad.com/viola-jones-method/

# the example from .xml file (one feature):
# <features>
#  <_>
#       <rects>
#         <_>
#            6 4 12 9 -1.</_>
#         <_>
#            6 7 12 3 3.</_></rects></_>
# </features>


class HaarFeature:

    def __init__(self, rect_list):
        self.rect_list = rect_list

    def cacl_feature(self, cumsum_marix):
        f_x = 0
        for rects in self.rect_list:
            if (rects.w, rects.h) >= MAXSIZE:
                continue
            else:
                f_x += (cumsum_marix[rects.y - 1 + rects.h][rects.x - 1 + rects.w]
                        - cumsum_marix[rects.y - 1][rects.x - 1 + rects.w]
                        - cumsum_marix[rects.y - 1 + rects.h][rects.x - 1]
                        + cumsum_marix[rects.y - 1][rects.x - 1]) * rects.weight
        return f_x




PATH = '/content/drive/My Drive/КУ Курсачи/Курсовой проект 2020/files/'
HaarRect = namedtuple('HaarRect', 'x y w h weight')

# function parse_file reads an input .xml file
# and returns list with HaarRect

def parse_file():
    h_features = []
    # загрузим xml и распарсим его
    with open(os.path.join(PATH, 'my_features.xml')) as fobj:
        xml = fobj.read()
        root = etree.fromstring(xml)

        for elems in root.getchildren():
            for rects in elems.getchildren():
                rectangles = []
                for haar_rect in rects.getchildren():
                    numbers = haar_rect.text[11:].split(" ")
                    rectangles.append(HaarRect(int(numbers[0]), int(numbers[1]), int(
                        numbers[2]), int(numbers[3]), float(numbers[4])))

                h_features.append(HaarFeature(rectangles))
    return h_features


RANDOM_SEED = 42

# function split_data separates input data
# into train and test samples


def split_data(faces_dataset):
    random.seed(RANDOM_SEED)
    random.shuffle(faces_dataset)

    y = np.array(list(map(lambda x: int(x.is_face), faces_dataset)))
    x_train, x_test, y_train, y_test = sk.train_test_split(faces_dataset, y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test


# function feature_creating creates haars and matrix features

def feature_creating(sample, feature_type, h_features):
    features = []

    for images in sample:
        img = np.array(images.img, 'uint8')
        if feature_type != 'hf':
            features.append(img.flatten())
        else:
            # integral representation of the image
            matrix = np.cumsum(np.cumsum(img, axis=0), axis=1)
            f_for_one_img = []
            for fichi in h_features:
                f_for_one_img.append(fichi.cacl_feature(matrix))
            features.append(f_for_one_img)
    return np.array(features)


col_list = ['logreg_', 'tree_', 'knn_', 'svm_', 'randforest_', 'ada_boost_', 'grad_boost_']
categories = ['hf_train', 'mf_train', 'pca_train', 'hf_test', 'mf_test', 'pca_test']
PATH = '/content/drive/My Drive/КУ Курсачи/Курсовой проект 2020/files/'


# function save_data saves all created data
def save_data(all_features, x_train, x_test, y_test, y_train):
    joblib.dump(all_features, os.path.join(PATH, 'all_features.pkl'))
    joblib.dump(y_train, os.path.join(PATH, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(PATH, 'y_test.pkl'))
    joblib.dump(x_train, os.path.join(PATH, 'x_train.pkl'))
    joblib.dump(x_test, os.path.join(PATH, 'x_test.pkl'))

# function final_data_preprocessing is the most important function in features.py
# it splits dataset, creates features and gives the output for the next stage


def final_data_preprocessing(faces_dataset, time_df):
    all_features = []
    h_features = parse_file()
    x_train, x_test, y_train, y_test = split_data(faces_dataset)
    samples = [x_train, x_test]

    pca = PCA(n_components=len(h_features))
    scaler = StandardScaler()

    for i, f_samples in enumerate(categories):
        parsed_str = f_samples.split("_")
        if parsed_str[1] == 'train':
            index = 0
        else:
            index = 1

        t0 = time.time()
        all_features.append(feature_creating(samples[index], parsed_str[0], h_features))
        t = time.time()
        time_df.loc[categories[i]] = (t - t0) / len(samples[index])

    # scale new features
    for i in range(0, len(all_features) // 2 - 1):
        all_features[i] = scaler.fit_transform(all_features[i])
        all_features[i + 3] = scaler.transform(all_features[i + 3])
    all_features[2] = pca.fit_transform(all_features[2])
    all_features[5] = pca.transform(all_features[5])
    save_data(all_features, x_train, x_test, y_test, y_train)


# function save_time saves time for feature creation


def save_time(time_df):
    time_of_features_creat = time_df[col_list[0]].values.tolist()
    joblib.dump(time_of_features_creat, os.path.join(PATH, 'time_of_features_creat.pkl'))
    joblib.dump(time_df, os.path.join(PATH, 'Dataframes', 'time_df_part0.pkl'))

# function create_features_and_return_time is created to run
# the process and show dataframe with time of feature creating to a user


def create_features_and_return_time(faces_dataset):
    time_df = pd.DataFrame(index=categories, columns=col_list)
    final_data_preprocessing(faces_dataset, time_df)
    save_time(time_df)
    return time_df


if __name__ == '__main__':
    haars_features = parse_file()
    print(haars_features[0].rect_list)
