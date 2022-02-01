import os
import random
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, RandomSampler, Sampler

# reproducible setup for testing
seed = 42
random.seed(seed)
np.random.seed(seed)


def _dataloader_init_fn():
    np.random.seed(seed)


def get_mnist_dataset(is_train_dataset: bool = True) -> MNIST:
    """
    Prepare MNIST dataset
    """
    return MNIST(
        os.getcwd(),
        download=True,
        transform=transforms.ToTensor(),
        train=is_train_dataset,
    )


def get_loader(is_train_set: bool = True) -> DataLoader:
    """
    Prepare MNIST train dataset loader
    """
    _dataset = get_mnist_dataset(is_train_dataset=is_train_set)
    _dataset_sampler = RandomSampler(_dataset)
    return _get_loader(dataset=_dataset, dataset_sampler=_dataset_sampler)


def _get_loader(dataset: MNIST, dataset_sampler: Sampler) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=10,
        sampler=dataset_sampler,
        worker_init_fn=_dataloader_init_fn,
    )
