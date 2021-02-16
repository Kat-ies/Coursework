"""
functions:
make_images_dict(directory=IMG_PATH, max_dict_size=9000, is_color=False)
make_frames_dict(image_dict)
"""

import os
from PIL import Image
from collections import namedtuple

PATH = '/content/drive/My Drive/КУ Курсачи/Курсовой проект 2020/files/'
IMG_PATH = 'WIDER_train/images'


def make_images_dict(directory=IMG_PATH, max_dict_size=9000, is_color=False):
    """
    function make_image_dict creates a fixed-size dictionary of images.
    images are taken from IMG_PATH or from another path
    dict_images = {'file_name' : PIL Image}
    """
    list_name = {}
    list_dirs = os.listdir(directory)
    list_dirs.sort()
    for cur_dir in list_dirs:
        list_file = os.listdir(os.path.join(directory, cur_dir))
        list_file.sort()
        for cur_file in list_file:
            if not is_color:
                img = Image.open(os.path.join(
                    directory, cur_dir, cur_file)).convert("L")
            else:
                img = Image.open(os.path.join(directory, cur_dir, cur_file))
            list_name[cur_file] = img
            if len(list_name) == max_dict_size:
                return list_name


# rectangle params
Rectangle = namedtuple('Rectangle', 'x y w h')


def make_frames_dict(image_dict):
    """
    function make_frames_dict creates a dictionary of frames.
    each frame is a namedtuple
    frames will be loaded only for files from image_dict,
    other marked up data will be ignored
    dict_frames = {'file_name' : Rectangle}

    the format of txt input file:
    File name
    Number of bounding box
    x1, y1, w, h, blur, expression, illumination,   invalid, occlusion, pose

    the format for images without frames
    File name
    0
    0 0 0 0 0 0 0 0 0 0
    """
    dict_frames = {}  # dict of Rectangle
    # open file with information of marked up data
    with open(os.path.join(PATH, 'wider_face_train_bbx_gt.txt'), "r+") as file:
        for lines in file:
            # filename
            filename = lines.rstrip(' \n').split("/")[1]
            # number of rectangles
            n = int(file.readline())
            rects = []
            # if there are no rectangles, then
            # next line will be 0 0 0 0 0 0 0 and we also need to read it
            if n == 0:
                n += 1
            for elements in range(n):
                # for each rectangle, we load all the suggested data
                params = file.readline().rstrip(' \n').split(" ")
                params = params[:4]
                rects.append(Rectangle(int(params[0]), int(
                    params[1]), int(params[2]), int(params[3])))
            # if we have image with name filename
            if filename in image_dict:
                dict_frames[filename] = rects
    return dict_frames


if __name__ == '__main__':
    dict_images = make_images_dict()
    frames = make_frames_dict(dict_images)
    keys_list = list(dict_images.keys())
    print(keys_list[0])
    print(frames[keys_list[0]])
