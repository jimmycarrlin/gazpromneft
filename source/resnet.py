from functools import partial

from tqdm.notebook import tqdm, trange

import matplotlib

import torch
from torch.nn import BatchNorm2d, AdaptiveAvgPool2d, Linear
from torch import nn
from torch.nn.functional import softmax

from torchmetrics import Accuracy
from torchmetrics.classification.stat_scores import StatScores

from source.plotting import plot_random_preds
from source.context_managers import eval_mode, train_mode
from source.callback import DefaultCallback


# Aliases
Conv7x7 = partial(nn.Conv2d, kernel_size=7, padding=3, bias=False)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1, bias=False)
Conv1x1 = partial(nn.Conv2d, kernel_size=1, padding=0, bias=False)
MaxPool3x3 = partial(nn.MaxPool2d, kernel_size=3, padding=1)
ReLU = partial(nn.ReLU, inplace=True)

etqdm = partial(trange, unit="epoch", desc="Epoch loop")
btqdm = partial(tqdm, unit="batch", desc="Batch loop", leave=False)


class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    expansion = 1

    def __init__(self, in_channels, hid_channels, stride=1):
        super().__init__()

        self.conv1 = Conv3x3(in_channels, hid_channels, stride=stride)
        self.bn1 = BatchNorm2d(hid_channels)
        self.conv2 = Conv3x3(hid_channels, hid_channels, stride=1)
        self.bn2 = BatchNorm2d(hid_channels)
        self.relu = ReLU()

        if (stride != 1) or (in_channels != self.expansion * hid_channels):
            self.shortcut = nn.Sequential(
                Conv1x1(in_channels, self.expansion * hid_channels, stride=stride)
            )
        else:
            self.shortcut = torch.nn.Sequential()  # Identity

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet."""
    expansion = 4

    def __init__(self, in_channels, hid_channels, stride=1):
        super().__init__()

        self.conv1 = Conv1x1(in_channels, hid_channels, stride=1)
        self.bn1 = BatchNorm2d(hid_channels)
        self.conv2 = Conv3x3(hid_channels, hid_channels, stride=stride)
        self.bn2 = BatchNorm2d(hid_channels)
        self.conv3 = Conv1x1(hid_channels, hid_channels * self.expansion, stride=1)
        self.bn3 = BatchNorm2d(hid_channels * self.expansion)
        self.relu = ReLU()

        if (stride != 1) or (in_channels != self.expansion * hid_channels):
            self.shortcut = nn.Sequential(
                Conv1x1(in_channels, self.expansion * hid_channels, stride=stride)
            )
        else:
            self.shortcut = nn.Sequential()  # Identity

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block_cls, layers, num_classes):
        super().__init__()
        self.block_cls = block_cls
        self.layers = layers
        self.num_classes = num_classes
        self.in_channels = 64

        self.conv1 = Conv7x7(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool3x3(stride=2)
        self.layer1 = self._make_layer(block_cls, 64, layers[0], first_stride=1)
        self.layer2 = self._make_layer(block_cls, 128, layers[1], first_stride=2)
        self.layer3 = self._make_layer(block_cls, 256, layers[2], first_stride=2)
        self.layer4 = self._make_layer(block_cls, 512, layers[3], first_stride=2)
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(512 * block_cls.expansion, num_classes)

    def _make_layer(self, block_cls, hid_channels, num_blocks, first_stride):
        layer = nn.Sequential()

        layer.add_module(
            '0', block_cls(self.in_channels, hid_channels, first_stride)
        )
        self.in_channels = hid_channels * block_cls.expansion

        for i in range(1, num_blocks):
            layer.add_module(
                str(i), block_cls(self.in_channels, hid_channels, stride=1)
            )

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def extra_repr(self):
        num_layers = sum(self.layers) * (2 if self.block_cls == BasicBlock else 3) + 2
        return f"{self._get_name()}{num_layers}"


# Standard architectures
def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(BasicBlock, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(BasicBlock, [3, 8, 36, 3], num_classes)


class TorchModel:
    """Class wrapping torch models."""
    def __init__(self, model, optimizer, criterion, metrics=None, callback=DefaultCallback()):
        if hasattr(criterion, "reduction"):
            criterion.reduction = "sum"

        if metrics is None:
            metrics = [Accuracy(top_k=1)]
        else:
            for metric in metrics:
                if hasattr(metric, "top_k"):
                    metric.top_k = metric.top_k or 1
                else:
                    raise TypeError

        self.optimizer = optimizer
        self.criterion = criterion
        self.callback = callback

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        self.model = self.to_device(model)
        self.metrics = self.to_device(metrics)

    def train(self, trainloader, valloader, epochs=1):
        with self.callback as callback:

            for epoch in etqdm(epochs):
                train_info = self._train(trainloader)
                val_info = self._validate(valloader)

                callback(self, train_info, val_info, epoch)

        return train_info, val_info

    def _train(self, loader):
        with train_mode(self.model) as model:

            TD, FD = partial(TorchModel.to_device, self), partial(TorchModel.from_device, self)  # Aliases

            epoch_loss = TD(torch.zeros(1))
            for input, labels in btqdm(loader):
                input, labels = TD((input, labels))
                model.zero_grad()
                output = model(input)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                probs = softmax(output, dim=1)

                epoch_loss += loss
                self._update_metrics(probs, labels)

            epoch_loss /= len(loader.dataset)
            metrics = self._compute_and_reset_metrics()

        return FD((epoch_loss, metrics))

    def _validate(self, loader):

        with eval_mode(self.model) as model:

            TD, FD = partial(TorchModel.to_device, self), partial(TorchModel.from_device, self)  # Aliases

            loss = TD(torch.zeros(1))
            for input, labels in loader:
                input, labels = TD((input, labels))
                output = model(input)
                probs = softmax(output, dim=1)

                loss += self.criterion(output, labels)
                self._update_metrics(probs, labels)

            loss /= TD(len(loader.dataset))
            metrics = self._compute_and_reset_metrics()
            random_preds = plot_random_preds(self, loader)

        return FD((loss, metrics, random_preds))

    def _update_metrics(self, probs, labels):
        for metric in self.metrics:
            metric.update(probs, labels)

    def _reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def _compute_and_reset_metrics(self):
        metrics_values = [metric.compute() for metric in self.metrics]
        self._reset_metrics()

        return metrics_values

    def test(self, testloader):
        return self._validate(testloader)

    def predict_proba(self, input):
        with eval_mode(self.model) as model:
            output = model(input)
            probs = softmax(output, dim=1)

        return probs

    def predict(self, input):
        with eval_mode(self.model) as model:
            output = model(input)
            _, preds = torch.max(output, dim=1)

        return preds

    def to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return type(obj)(self.to_device(x) for x in obj)
        if isinstance(obj, (torch.Tensor, torch.nn.Module, StatScores)):
            return obj.to(self.device)
        if isinstance(obj, (int, float)):
            return torch.tensor(obj).to(self.device)
        else:
            raise TypeError(f"Got: {type(obj)}")

    def from_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return type(obj)(self.from_device(x) for x in obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, matplotlib.figure.Figure):
            return obj
        else:
            raise TypeError(f"Got: {type(obj)}")