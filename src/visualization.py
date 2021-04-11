"""
functions:
show_wider_face_dataset(images_dict, frames_dict)
show_faces_dataset(faces_dataset)
plotting(dataframe, feature_type, index, col_list, fig, ax)
show_wrong_classified_obj()
plot_graphic(y_points, title)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from PIL import ImageDraw
from constants import *
from data_loader import load_sample, load_dataframe


def show_wider_face_dataset(images_dict, frames_dict):
    """
    function show_wider_face_dataset(images_dict, frames_dict)
    shows some visual examples from original dataset
    for making sure, that data was correctly downloaded and marked
    """
    fig = plt.figure(figsize=(30, 40))
    cols = COLUMNS_FOR_WF_DATASET
    rows = ROWS_FOR_WF_DATASET

    # ax enables access to manipulate each of subplots
    ax = []
    np.random.seed(RANDOM_SEED)

    keys_list = list(images_dict.keys())

    for i in range(cols * rows):
        rand_num = np.random.randint(0, len(images_dict))

        key = keys_list[rand_num]
        image = images_dict[key]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, cols, i + 1))
        for rect in frames_dict[key]:
            rect = patches.Rectangle(
                (rect.x, rect.y), rect.w, rect.h, linewidth=2, edgecolor='r', facecolor='none')
            ax[i].add_patch(rect)
        ax[-1].set_title(key)  # set title
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.show()  # finally, render the plot


def show_faces_dataset(faces_dataset):
    """
    function show_faces_dataset(faces_dataset)
    shows some visual examples from new dataset
    for making sure, that dataset creation was successful
    """
    fig = plt.figure(figsize=(10, 15))
    cols = COLUMNS_FOR_NEW_DATASET
    rows = ROWS_FOR_NEW_DATASET

    # ax enables access to manipulate each of subplots
    ax = []
    np.random.seed(RANDOM_SEED)

    for i in range(cols * rows):
        rand_num = np.random.randint(0, len(faces_dataset))
        image = faces_dataset[rand_num].img
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, cols, i + 1))
        if faces_dataset[rand_num].is_face == 1:
            ax[-1].set_title('face')  # set title
        else:
            ax[-1].set_title('not face')
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.show()  # finally, render the plot


def plotting(dataframe, feature_type, index, col_list, fig, ax,  x_test):
    """
    function plotting(dataframe, feature_type, index, col_list, fig, ax)
    is a support function for show_wrong_classified_obj()
    """
    for i in range(len(col_list)):
        for j in range(COLUMNS_FOR_WRONG_CLF_OBJ):
            rand_num = np.random.randint(0, len(dataframe))

            while dataframe.iloc[rand_num][col_list[i]] == \
                    dataframe.iloc[rand_num]['y_test']:
                rand_num = np.random.randint(0, len(dataframe))

            image = x_test[rand_num].img
            ax.append(fig.add_subplot(ROWS_FOR_WRONG_CLF_OBJ,
                                      COLUMNS_FOR_WRONG_CLF_OBJ, index))
            index += 1

            if dataframe.iloc[rand_num]['y_test'] == 0:
                ax[-1].set_title(feature_type + '; ' + col_list[i] + '; not a face')
            else:
                ax[-1].set_title(feature_type + '; ' + col_list[i]
                                 + '; isn\'t detected ')

            plt.imshow(image, cmap='gray', vmin=0, vmax=255)


def show_wrong_classified_obj():
    """
    function show_wrong_classified_obj() shows some examples
    ml-models gave false positive or false negative detections
    """
    indexes = np.arange(1, 96, 32)
    col_list = COL_LIST
    col_list.append('voting (cols [1-5) )')
    x_test = load_sample('x_test')

    fig = plt.figure(figsize=(8, 60))
    ax = []
    plt.rcParams.update({'font.size': 6})
    np.random.seed(RANDOM_SEED)

    plotting(load_dataframe('haars_df'), 'hf', indexes[0],
             col_list, fig, ax,  x_test)
    plotting(load_dataframe('matrix_df'), 'mf', indexes[1],
             col_list, fig, ax,  x_test)
    plotting(load_dataframe('matrix_pca_df'), 'mf_pca', indexes[2],
             col_list, fig, ax,  x_test)

    plt.show()


def plot_graphic(y_points, title):
    """
    function plot_graphic(y_points, title) plots
    graphics for neural networks training part
    """
    x_points = lambda x: np.arange(0, len(y_points[x]))

    plt.figure(figsize=(20, 10))
    plt.title(title, fontsize=18, fontname='Times New Roman')
    plt.xlabel('Epoch', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Score', fontsize=16, fontname='Times New Roman')
    plt.plot(x_points(0), y_points[0], color='#fb607f', linestyle='-')
    if len(y_points) > 1:
        plt.plot(x_points(1), y_points[1], color='#906bff', linestyle='-')
        plt.plot(x_points(2), y_points[2], color='#c71585', linestyle='-')
        plt.legend(FEATURES_LIST, loc='center', shadow=True, fontsize=18)
    else:
        plt.legend(['matrix'], loc='center', shadow=True, fontsize=18)
    plt.show()
