import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from src.models.base import Model
import torchmetrics
from torch.cuda.amp import autocast

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics import MAP
from src.config import settings as st
from src.utils.general import config_to_bool

# from torchmetrics.detection.mean_ap import MeanAveragePrecision

# %%


def get_model_instance_segmentation():

    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
    )

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    NUM_CLASSES = 91
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model


class FasterRCNNModel(Model):
    def __init__(self):

        init_values = {
            "model": get_model_instance_segmentation(),
            "optimizer": torch.optim.SGD,
            "optimizer_args": {
                "lr": 0.005,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "criterion": nn.BCEWithLogitsLoss,
            "metric": MAP,
        }

        super().__init__(**init_values)

    def train(self, dataloader, epoch, metric_logger, **kwargs):

        model = self.init_model
        optimizer = self.init_optimizer
        criterion = self.init_criterion
        device = self.device

        metric = self.metric()
        metric.to(device)

        running_loss = 0
        samples = 0

        model.train()

        cutoff = kwargs.get("cutoff")

        for i, data in metric_logger.log_round(dataloader, epoch, **kwargs):

            samples += len(data[0])
            metric_logger.accumulate_samples(epoch, len(data[0]))

            if not kwargs["automatic_gpu_transfer"]:
                images = list(image.to(device) for image in data[0])
                targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

            if config_to_bool(st.is_cutoff_run_model):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                running_loss += loss_value

                optimizer.zero_grad()

                losses.backward()
                optimizer.step()

            # clear gradients for this training step

        running_loss /= samples
        metric_logger.log_loss(running_loss, "train")

    def evaluate(self, dataloader, epoch, mode, metric_logger, **kwargs):

        model = self.init_model
        model.eval()

        optimizer = self.init_optimizer
        criterion = self.init_criterion
        device = self.device

        metric = self.metric()
        metric.to(device)

        running_loss = 0
        samples = 0

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print(f"Epoch: {epoch}, iteration: {i}", end="\r", flush=True)
                samples += len(data[0])

                if not kwargs["automatic_gpu_transfer"]:
                    images = list(image.to(device) for image in data[0])
                    targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

                model.eval()
                outputs = model(images)

                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                running_loss += loss_value

                p = [{k: v.to(device) for k, v in p_.items()} for p_ in outputs]
                t = [{k: v.to(device) for k, v in t_.items()} for t_ in targets]
                metric.update(p, t)

        m = metric.compute()
        m = {k: v.item() for k, v in m.items()}

        running_loss /= samples
        metric_logger.log_loss(running_loss, mode)
        metric_logger.log_metric(m, mode)
