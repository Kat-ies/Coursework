from face_detection.bounding_box import BoundingBox
from face_detection.utils_for_mAP import CoordinatesType, BBType, BBFormat
from constants import *
from face_detection.transforms import test_transforms
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np


def add_boxes(test_dicts, my_bounding_boxes, model):
    dict_images = test_dicts[0]
    dict_frames = test_dicts[1]

    for key, rects in dict_frames.items():

        # let's add ground_truth_boxes first
        for rectangle in rects:
            x, y, w, h = rectangle
            gt_bounding_box = BoundingBox(imageName=key, classId='face', x=x, y=y,
                                          w=w, h=h, typeCoordinates=CoordinatesType.Absolute,
                                          bbType=BBType.GroundTruth, format=BBFormat.XYWH,
                                          imgSize=dict_images[key].size)
            my_bounding_boxes.addBoundingBox(gt_bounding_box)

        # now let's make predictions and add to our boxes all detected boxes
        image = test_transforms(dict_images[key])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with torch.no_grad():
            prediction = model([image.to(device)])

        boxes = list(prediction[0]['boxes'].cpu().numpy())
        scores = list(prediction[0]['scores'].cpu().numpy())

        for i, box in enumerate(boxes):
            x, y, x2, y2 = box
            detected_bounding_box = BoundingBox(imageName=key, classId='face', classConfidence=scores[i],
                                                x=x, y=y, w=x2, h=y2, typeCoordinates=CoordinatesType.Absolute,
                                                bbType=BBType.Detected, format=BBFormat.XYX2Y2,
                                                imgSize=dict_images[key].size)
            my_bounding_boxes.addBoundingBox(detected_bounding_box)


def show_predictions(images, predictions, threshold):
    fig = plt.figure(figsize=(60, 70))
    cols = 1
    rows = len(images)

    # ax enables access to manipulate each of subplots
    ax = []
    i = 0

    for image, prediction in zip(images, predictions):

        boxes = list(prediction[0]['boxes'].cpu().numpy())
        scores = list(prediction[0]['scores'].cpu().numpy())

        ax.append(fig.add_subplot(rows, cols, i + 1))
        i += 1

        for box, score in zip(boxes, scores):

            if score < threshold:
                continue

            img1 = ImageDraw.Draw(image)
            img1.rectangle(box, fill=None, outline="red")

            img2 = ImageDraw.Draw(image)
            img2.rectangle((box[0], box[1],
                            box[0] + SCORE_BACKGROUND_LABEL_WIDTH,
                            box[1] + SCORE_BACKGROUND_LABEL_HEIGHT),
                           fill='black', outline="black")

            img3 = ImageDraw.Draw(image)
            img3.text((box[0], box[1]), 'score: ' + f"{score:.{2}f}")

        plt.imshow(image)

    plt.show()  # finally, render the plot


def show_image_examples(detector, threshold=0.5, folder='images'):
    files = os.listdir(os.path.join(WORK_PATH, 'images'))
    files.sort()

    images = []
    predictions = []

    for file in files:
        image = Image.open(os.path.join(WORK_PATH, folder, file))
        images.append(image)
        predictions.append(detector.detect_at_one_image(image))

    show_predictions(images, predictions, threshold)


def plot_train_curves(y_points):
    """
    function plot_train_curves(y_points) plots
    graphics for training part
    """
    x_points = lambda x: np.arange(0, len(y_points[x]))

    plt.figure(figsize=(15, 8))
    plt.title('losses graphic', fontsize=18, fontname='Times New Roman')
    plt.xlabel('Epoch', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Loss', fontsize=16, fontname='Times New Roman')

    plt.plot(x_points(0), y_points[0], color='#fb607f', linestyle='-')
    plt.plot(x_points(1), y_points[1], color='#906bff', linestyle='-')
    plt.plot(x_points(2), y_points[2], color='#c71585', linestyle='-')
    plt.plot(x_points(3), y_points[3], color='#ea7500', linestyle='-')
    plt.plot(x_points(4), y_points[4], color='#015d52', linestyle='-')
    plt.legend(LOG_LOSSES, loc='right', shadow=True, fontsize=18)

    plt.show()
