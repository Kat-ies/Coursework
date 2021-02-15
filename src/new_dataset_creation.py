"""
functions:
dataset_create(images_dict, frames_dict)
"""

from PIL import Image
from collections import namedtuple

Faces = namedtuple('Faces', 'img is_face filename rect')
Rectangle = namedtuple('Rectangle', 'x y w h')

MAXSIZE = (25, 25)
MINSIZE = (20, 20)
random_box = (30, 30, 30, 30)


def dataset_create(images_dict, frames_dict):
    """
    function dataset_create(images_dict, frames_dict) makes a new marked-up dataset
    the wider_face dataset is not suitable for the face classification task
    to solve the classification problem,
    we want to input an image of a face or not a face
    so we will create such a dataset

    creation rules:

    1. if there is a face in the image, then we cut it out,
    for a non-face element we take a rectangle of the same size
    from the location diagonally from the face,
    i.e. the images have a common point

    2. if there is no face, then we take a random place as a non-face
    In our case it is random_box(300, 300, 300, 300)

    3. new images are reduced to a single size according
    to the Image.LANCZOS principle (it gives the best quality when
    we change the image (the k-nearest neighbor method is used as a basis))

    4. face images in wider_face are both small and very large, then
    the images with size less then MINSIZE will be ignored
    """
    faces = []
    for key in frames_dict:
        for rect in frames_dict[key]:
            if rect == (0, 0, 0, 0):
                img_not_face = images_dict[key].crop(
                    random_box).resize(MAXSIZE, Image.LANCZOS)
                faces.append(Faces(img_not_face, 0, key,
                                   Rectangle(random_box[0], random_box[1], random_box[2], random_box[3])))
            elif (rect.w, rect.h) < MINSIZE:
                continue
            else:
                img_face = images_dict[key].crop(
                    (rect.x, rect.y, rect.x + rect.w, rect.y + rect.h)).resize(MAXSIZE, Image.LANCZOS)
                img_not_face = images_dict[key].crop(
                    (rect.x + rect.w, rect.y + rect.h, rect.x + 2 * rect.w, rect.y + 2 * rect.h))\
                    .resize(MAXSIZE, Image.LANCZOS)
                faces.append(Faces(img_face, 1, key, rect))
                faces.append(Faces(img_not_face, 0, key,
                                   Rectangle(rect.x + rect.w, rect.y + rect.h, rect.w, rect.h)))
    return faces
