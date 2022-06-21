from contextlib import contextmanager


@contextmanager
def eval_mode(model):
    """Temporarily switch to evaluation mode."""
    is_train = model.training
    is_requires_grad = next(model.parameters()).requires_grad
    try:
        model.train(False)
        model.requires_grad_(False)
        yield model
    finally:
        model.train(is_train)
        model.requires_grad_(is_requires_grad)


@contextmanager
def train_mode(model):
    """Temporarily switch to train mode."""
    is_train = model.training
    is_requires_grad = next(model.parameters()).requires_grad
    try:
        model.train(True)
        model.requires_grad_(True)
        yield model
    finally:
        model.train(is_train)
        model.requires_grad_(is_requires_grad)