from data_saver import save_nn_model
from data_loader import load_nn_model
from face_detection.transforms import *
from face_detection.custom_dataset import FacesDataset
from face_detection.split_data import make_samples
from face_detection.utils import collate_fn
from face_detection.visualization import add_boxes, plot_train_curves
from torch.utils.data import DataLoader
import torch
from constants import *
from face_detection.bounding_boxes import BoundingBoxes
import face_detection.evaluator as eval
from face_detection.transforms import train_transforms
from face_detection.train_one_epoch import train_one_epoch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from PIL import Image, JpegImagePlugin
import cv2
import numpy as np
import os


class Detector:
    def __init__(self, model=None, model_name='detector', params=None,
                 lr_scheduler=None, optimizer=None, pretrained=True):
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.params = params
        self.optimizer = optimizer
        # learning rate scheduler which decreases the learning rate by 10x every 3 epochs
        self.lr_scheduler = lr_scheduler
        self.model_name = model_name
        self.pretrained = pretrained

    def train(self, num_epochs=5):
        # Let's fix everything that can be fixed to get the same results between runs
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        self.model.to(self.device)
        print(self.device)

        train_dicts, val_dicts = make_samples(mode='TRAIN_VAL', max_dictionary_size=2500)
        print('train dataset size: ', str(len(train_dicts[0])))

        train_dataset = FacesDataset(train_dicts, transforms=train_transforms)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn)

        metric_collector = []

        for epoch in range(num_epochs):
            self.model.train()
            metric_logger = train_one_epoch(self.model, self.optimizer,
                                            train_data_loader, self.device, epoch, print_freq=100)
            metric_collector.append(metric_logger)
            self.lr_scheduler.step()

            self.validate(val_dicts)

        losses = [[], [], [], [], []]

        # "{median:.4f} ({global_avg:.4f})" - формат вывода потерь
        for log in metric_collector:
            for i, key in enumerate(LOG_LOSSES):
                losses[i].append(log.meters[key].median)

        plot_train_curves(losses)
        save_nn_model(self.model.state_dict(), self.model_name, path=WORK_PATH, folder='Models')

    def validate(self, val_dict):
        self.model.eval()
        my_bounding_boxes = BoundingBoxes()
        add_boxes(val_dict, my_bounding_boxes, self.model)

        evaluator = eval.Evaluator()

        results = evaluator.PlotPrecisionRecallCurve(
            boundingBoxes=my_bounding_boxes, showGraphic=False,
            showInterpolatedPrecision=False, showAP=True)

        print('AP' + f": {results[0]['AP']}")

    def test(self):

        my_bounding_boxes = BoundingBoxes()

        test_dicts = make_samples(mode='TEST')

        self.model.load_state_dict(load_nn_model(self.model_name, path=WORK_PATH, folder='Models'))
        self.model.eval()
        self.model.to(self.device)

        add_boxes(test_dicts, my_bounding_boxes, self.model)

        evaluator = eval.Evaluator()

        results = evaluator.PlotPrecisionRecallCurve(
            boundingBoxes=my_bounding_boxes, showGraphic=False,
            showInterpolatedPrecision=False, showAP=True)

        info = ['AP', 'total positives', 'total TP', 'total FP']

        for params in info:
            print(params, results[0][params])

    def detect_at_one_image(self, image):
        self.model.load_state_dict(load_nn_model(self.model_name, path=WORK_PATH, folder='Models'))
        self.model.eval()
        self.model.to(self.device)

        if not isinstance(image, JpegImagePlugin.JpegImageFile):
            image = Image.open(image)

        img = test_transforms(image)
        with torch.no_grad():
            return self.model([img.to(self.device)])

    def __repr__(self):
        return self.model


class FasterRCNNDetector(Detector):

    def __init__(self, trained=True, pretrained=True):
        super().__init__(model=self.fine_tune_model(pretrained),
                         model_name='faster_rcnn' if pretrained else 'faster_rcnn_not_pretrained')
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.trained = trained
        self.pretrained = pretrained
        self.optimizer = torch.optim.Adam(self.params, lr=0.0005, betas=(0.9, 0.999), weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def fine_tune_model(self, pretrained):
        num_classes = 2
        # load an object detection model pre-trained on COCO == pretrained=True
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model


class RetinaNetDetector(Detector):

    def __init__(self, trained=True, pretrained=True):
        super().__init__(model=self.fine_tune_model(pretrained),
                         model_name='retina_net' if pretrained else 'retina_net_not_pretrained')
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.trained = trained
        self.pretrained = pretrained
        self.optimizer = torch.optim.SGD(self.params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def fine_tune_model(self, pretrained):
        num_classes = 2

        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained)

        # get number of input features and anchor boxed for the classifier
        in_features = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors

        # replace the pre-trained head with a new one
        model.head = RetinaNetHead(in_features, num_anchors, num_classes)
        return model


class ViolaJhonesDetector:

    def __init__(self, model_name='viola_jhones', path=PROJECT_PATH, file='haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(os.path.join(path, file))
        self.model_name = model_name

    # по идее можно добавить ещё test, чтобы понять, насколько этот метод лучше моих моделек
    def detect_at_one_image(self, image):
        if not isinstance(image, JpegImagePlugin.JpegImageFile):
            image = Image.open(image)

        img = np.array(image.convert("L"), 'uint8')
        boxes = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

        # converting [x, y, w, h] boxes to (x0, y0, x1, y1)
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]

        scores = torch.ones(len(boxes))
        return [{'boxes': torch.tensor(boxes, dtype=torch.float32), 'scores': scores}]

    def __repr__(self):
        return self.model
