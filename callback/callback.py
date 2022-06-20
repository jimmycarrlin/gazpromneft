import torch

from abc import ABCMeta, abstractmethod, ABC
from collections.abc import Iterable

from torch.utils.tensorboard import SummaryWriter


class Callback(metaclass=ABCMeta):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def __call__(self, torchmodel, train_info, val_info, epoch):
        pass


class CompositeCallback(Callback, Iterable):
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
    def __init__(self, log_dir,
                 comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):

        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)

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

            self.add_scalar(f"{metric_name}Top{metric.top_k}/{mode}", val_info[1][i])

        self.add_figure("Random predictions", test_info[2])

class Profiler(torch.profiler.profile, Callback):
    def __call__(self, *_):
        self.step()

    @classmethod
    def make_default(cls, log_dir):
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=2)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(log_dir)
        profile_memory = True

        return cls(schedule=schedule, on_trace_ready=on_trace_ready, profile_memory=profile_memory)


class Saver(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.best_loss = float("inf")
        self.best_state = None

    def __call__(self, torchmodel, train_info, _, epoch):
        model_state_dict = torchmodel.model.state_dict()
        torch.save(model_state_dict, self.save_dir / "backup.pt")

        loss, *_ = train_info
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = model_state_dict
            torch.save(self.best_state, self.save_dir / "best.pt")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DefaultCallback(Callback):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, torchmodel, train_info, val_info, epoch):
        pass