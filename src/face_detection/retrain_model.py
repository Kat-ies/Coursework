from face_detection.train_one_epoch import train_one_epoch
from data_saver import save_nn_model
from data_loader import load_model
import math
from face_detection.transforms import *
from face_detection.custom_dataset import MyDataset
from face_detection.split_data import make_samples
from face_detection.utils import collate_fn
from face_detection.visualization import add_boxes
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from constants import *

from face_detection.bounding_box import BoundingBox
from face_detection.bounding_boxes import BoundingBoxes
from face_detection.utils_for_mAP import CoordinatesType, BBType, BBFormat
import face_detection.evaluator as eval
from face_detection.transforms import test_transforms


def get_object_detection_model():
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = 2  # 1 + background

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def set_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def validate(val_dict):
    my_bounding_boxes = BoundingBoxes()
    add_boxes(val_dict, my_bounding_boxes)

    evaluator = eval.Evaluator()

    results = evaluator.PlotPrecisionRecallCurve(
        boundingBoxes=my_bounding_boxes, showGraphic=False,
        showInterpolatedPrecision=False, showAP=True)

    print('AP' + f": {results[0]['AP']}")


def retrain_model(num_epochs=3, model='faster_rcnn'):
    device = set_device()
    model = load_model(trained=False, mode='ADMIN')
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_dicts, val_dicts = make_samples(mode='TRAIN_VAL')

    # Let's fix everything that can be fixed to get the same results between runs
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # train_dataset = MyDataset(load_dicts(train_dicts), transforms=train_transforms)
    train_dataset = MyDataset(train_dicts, transforms=train_transforms)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100)
        lr_scheduler.step()

        validate(val_dicts)

    save_nn_model(model.state_dict(), model, path=WORK_PATH, folder='Models')
