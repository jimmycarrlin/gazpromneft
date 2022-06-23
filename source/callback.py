import os

import torch

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ray import tune


class Callback(metaclass=ABCMeta):
    """Base class for all callbacks. Implement Composite pattern (паттерн компоновщик)."""
    @abstractmethod
    def __call__(self, torchmodel, train_info, val_info, epoch):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class CompositeCallback(Callback, Iterable):
    """Composite callback. Container for primitive callbacks."""
    def __init__(self, children=None):
        self.children = children if children else []

    def add(self, child):
        self.children.append(child)

    def remove(self, child):
        self.children.remove(child)

    def __call__(self, torchmodel, train_info, val_info, epoch):
        for child in self:
            child.__call__(torchmodel, train_info, val_info, epoch)

    def __enter__(self):
        for child in self:
            child.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for child in self:
            child.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return iter(self.children)


class ClassificationReporter(SummaryWriter, Callback):
    """Primitive callback. Report losses and metrics."""
    def __call__(self, torchmodel, train_info, val_info, epoch):
        self.add_scalar("Loss/train", train_info[0], epoch)
        self.add_scalar("Loss/val", val_info[0], epoch)

        for i, metric in enumerate(torchmodel.metrics):
            metric_name = type(metric).__name__

            self.add_scalar(f"{metric_name}Top{metric.top_k}/train", train_info[1][i], epoch)
            self.add_scalar(f"{metric_name}Top{metric.top_k}/val", val_info[1][i], epoch)

        self.add_figure("Random predictions", val_info[2], epoch)

    def report(self, torchmodel, test_info, mode="test"):
        self.add_scalar(f"Loss/{mode}", test_info[0])

        for i, metric in enumerate(torchmodel.metrics):
            metric_name = type(metric).__name__

            self.add_scalar(f"{metric_name}Top{metric.top_k}/{mode}", test_info[1][i])

        self.add_figure("Random predictions", test_info[2])


class Profiler(torch.profiler.profile, Callback):
    """Primitive callback. Profile CPU, GPU and memory usage."""
    def __call__(self, *_):
        self.step()

    @classmethod
    def make_default(cls, log_dir):
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=2)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(log_dir)
        profile_memory = True

        return cls(schedule=schedule, on_trace_ready=on_trace_ready, profile_memory=profile_memory)


class Saver(Callback):
    """Primitive callback. Save last and best models."""
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.best_train_loss = float("inf")
        self.best_train_state = None

        self.best_val_loss = float("inf")
        self.best_val_state = None

    def __call__(self, torchmodel, train_info, val_info, epoch):
        model = torchmodel.model
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save(model_state_dict, self.save_dir / "backup.pt")

        train_loss, *_ = train_info
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            self.best_train_state = model_state_dict
            torch.save(self.best_train_state, self.save_dir / "best_test.pt")

        val_loss, *_ = val_info
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_state = model_state_dict
            torch.save(self.best_val_state, self.save_dir / "best_val.pt")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Tuner(Callback):
    """Primitive callback. Used only in parallel with ray tune."""
    def __call__(self, torchmodel, train_info, val_info, epoch):
        with tune.checkpoint_dir(epoch) as checkpoint_dir:

            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(torchmodel.model.state_dict(), torchmodel.optimizer.state_dict(), path)

        tune.report(loss=val_info[0], accuracy=val_info[1][0])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DefaultCallback(Callback):
    """Primitive callback. Do nothing."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, torchmodel, train_info, val_info, epoch):
        pass