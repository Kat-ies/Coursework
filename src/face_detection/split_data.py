from feature_creation.images_and_frames import *
from data_loader import unpacking_zips
from face_detection.utils import collate_fn
import sklearn.model_selection as sk
import random
from constants import *


def split_dictionaries(dict_frames, dict_images, coef=0.25):
    size = int(len(dict_images) * coef)  # val, test sizes
    train_size = len(dict_images) - 2 * size

    test_dict_frames, train_dict_frames, val_dict_frames = {}, {}, {}
    test_dict_images, train_dict_images, val_dict_images = {}, {}, {}

    keys = list(dict_images.keys())

    for i, key in enumerate(keys):
        if i < train_size:
            train_dict_frames[key] = dict_frames[key]
            train_dict_images[key] = dict_images[key]
        elif i < train_size + size:
            test_dict_frames[key] = dict_frames[key]
            test_dict_images[key] = dict_images[key]
        else:
            val_dict_frames[key] = dict_frames[key]
            val_dict_images[key] = dict_images[key]

    # free memory
    dict_frames.clear(), dict_images.clear()

    return (test_dict_images, test_dict_frames), (train_dict_images, train_dict_frames),\
           (val_dict_images, val_dict_frames)


def make_valide_dict(dict_images, dict_frames):
    # здесь начинаются небольшие танцы с бубном, тк нейросеть
    # не принимает рамки вырожденные в точку или в прямую
    keys_list = list(dict_frames.keys())
    for key in keys_list:
        for rects in dict_frames[key]:
            if rects == Rectangle(0, 0, 0, 0) or rects.w == 0 or rects.h == 0:
                if key in dict_frames:
                    del dict_frames[key]
                if key in dict_images:
                    del dict_images[key]


def make_samples(mode='TRAIN_VAL', max_dict_size=4000):
    unpacking_zips()

    try:
        # поскольку картинки и рамки я уже когда-то юзала, то можно снова взять эти функции
        dict_images = make_images_dict(is_color=True, max_dict_size=max_dict_size)
        print(len(dict_images))
        dict_frames = make_frames_dict(dict_images)

        make_valide_dict(dict_images, dict_frames)
        test_dicts, train_dicts, val_dicts = split_dictionaries(dict_frames, dict_images)

        # я хотела сохранить словари, чтобы больше не грузить исходный датасет,
        # а сразу грузить их, но оно почему вываливается по памяти :(
        # save_dicts(train_dicts, 'train_dicts')
        # save_dicts(test_dicts, 'test_dicts')

        if mode == 'TRAIN_VAL':
            return train_dicts, val_dicts
        elif mode == 'TRAIN':
            return train_dicts
        elif mode == 'TEST':
            return test_dicts
        elif mode == 'VAL':
            return val_dicts
        else:
            raise RuntimeError('Invalid working mode')

    except RuntimeError as re:
        print(*re.args)
