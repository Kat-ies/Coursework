"""
functions:
show_wider_face_dataset(images_dict, frames_dict)
show_faces_dataset(faces_dataset)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from PIL import ImageDraw
from constants import *


def show_wider_face_dataset(images_dict, frames_dict):
    """
    function show_wider_face_dataset
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
    function show_faces_dataset shows some visual examples from new dataset
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
