from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FacesDataset(Dataset):
    def __init__(self, train_dicts, transforms=None):
        self.images = train_dicts[0]
        self.frames = train_dicts[1]
        self.keys_list = list(train_dicts[0].keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        dict_key = self.keys_list[idx]
        img = self.images[dict_key]

        boxes = []
        labels = np.ones(len(self.frames[dict_key]))

        for rects in self.frames[dict_key]:
            boxes.append([rects.x, rects.y, rects.x + rects.w, rects.y + rects.h])

        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            # Note that target (including bbox) is also transformed\enhanced here,
            # which is different from transforms from torchvision import
            # Https://github.com/pytorch/vision/tree/master/references/detectionOfTransforms.py
            # There are examples of target transformations when RandomHorizontalFlip img,
            # target = self.transforms(img, target)
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)
