"""Adapted from the PyTorch Lightning quickstart example.

Source: https://pytorchlightning.ai/ (2021/02/04)
"""

# %%

import torch
from torch import nn
from torch.nn import functional as F
from src.models.base import Model
import torchmetrics
from torch.cuda.amp import autocast


from torch import optim
from torchvision.models import resnet18, ResNet18_Weights
from src.config import settings as st
from src.utils.general import config_to_bool


# %%


class CNNModel(Model):
    def __init__(self):

        init_values = {
            "model": resnet18(),
            "optimizer": optim.Adam,
            "optimizer_args": {"lr": 0.05},
            "criterion": nn.CrossEntropyLoss,
            "metric": torchmetrics.Accuracy,
        }

        super().__init__(**init_values)

    def train(self, dataloader, epoch, metric_logger, **kwargs):

        model = self.init_model
        optimizer = self.init_optimizer
        criterion = self.init_criterion
        device = self.device

        metric = self.metric()
        metric = metric.to(device)

        running_loss = 0
        samples = 0

        cutoff = kwargs.get("cutoff")

        for i, obj in metric_logger.log_round(dataloader, epoch, **kwargs):
            # For Deep Lake Dataloader
            if isinstance(obj, dict):
                inputs, labels = obj["images"], obj["labels"]
            else:
                inputs, labels = obj

            samples += inputs.shape[0]
            metric_logger.accumulate_samples(epoch, inputs.shape[0])
            if not kwargs["automatic_gpu_transfer"]:
                inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()

            if config_to_bool(st.is_cutoff_run_model):
                with autocast():
                    output = model(inputs)
                    loss = criterion(output, labels)

                # clear gradients for this training step
                optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()
                running_loss += loss.item()

                # apply gradients
                optimizer.step()

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

        with torch.no_grad():
            for obj in dataloader:
                # For Deep Lake Dataloader
                if isinstance(obj, dict):
                    data, target = obj["images"], obj["labels"]
                else:
                    data, target = obj

                samples += data.shape[0]
                data, target = data.to(device), target.to(device)
                target = target.squeeze()

                with autocast():
                    output = model(data)
                    loss = criterion(output, target)

                acc = metric(output, target)
                running_loss += loss.item()

        m = metric.compute()

        running_loss /= samples
        metric_logger.log_loss(running_loss, mode)
        metric_logger.log_metric({"accuracy": m.item()}, mode)
