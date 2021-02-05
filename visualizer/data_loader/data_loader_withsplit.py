import logging
import typing

import gin
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data._utils.collate import default_collate

LOGGER = logging.getLogger(__name__)


def split(
    dataset: torch.utils.data.Dataset,
    train: float = 0.7,
    valid: float = 0.2,
    test: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 0,
) -> typing.Tuple[SubsetRandomSampler, SubsetRandomSampler, SubsetRandomSampler]:
    """
    Function to split Dataset into train, test & valid SubsetRandomSamplers.
    This doesnot return in 3 seperate torch.utils.data.Dataset() classes but returns 3 different Samplers.

    :param dataset: torch.utils.data.Dataset()
        Dataset class e.g. GiniDataset()
    :param train: float
        Train data ratio, should be <1.0
    :param valid: float
        Validation data ratio, should be <1.0
    :param test: float
        Test data ratio, should be <1.0
    :param shuffle: bool
        Whether the output samplers will be shuffled or not.
    :param random_seed: int
        For setting np.random.seed() for shuffling (if shuffle==True)
    :return:
        train_sampler, valid_sampler, test_sampler: Tuple[SubsetRandomSampler, SubsetRandomSampler, SubsetRandomSampler]
        Individual samplers for train, validation and test data.
    """
    assert (
        int(train + valid + test) == 1
    ), "Train, validation and test set ratio do not add up to 1."
    num_data = len(dataset)
    indices = list(range(num_data))
    split_train = int(np.floor(train * num_data))
    split_valid = split_train + int(np.floor(valid * num_data))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = (
        indices[:split_train],
        indices[split_train:split_valid],
        indices[split_valid:],
    )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    return train_sampler, valid_sampler, test_sampler


@gin.configurable
def get_train_valid_test_loader_withsplit(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 8,
    random_seed: int = 0,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 1,
    pin_memory: bool = True,
    collate_fn: typing.Callable = default_collate,
) -> typing.Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    :param dataset: torch.utils.data.Dataset()
        Dataset class e.g. BraTSDataset()
    :param batch_size: int
        Batch size for training. The loader provides batch_size number of samples for each training time step.
    :param random_seed: int
        For setting np.random.seed() for shuffling (if shuffle==True)
    :param valid_size: float
        Validation data ratio, should be <1.0
    :param test_size: float
        Test data ratio, should be <1.0
    :param shuffle: bool
        If set true, the data is shuffled before sampling
    :param num_workers:: int
        How many subprocesses to use for data loading. 0 means the data will be loaded in the main process only.
    :param pin_memory: bool
        Decided whether data loader should copy the Tensors into CUDA pinned memory.
    :return:
        train_loader, valid_loader, test_loader:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        DataLoader for each sets of data.
    """

    # load dataset
    # get train, validation and test samplers
    train_sampler, valid_sampler, test_sampler = split(
        dataset,
        train=1.0 - valid_size - test_size,
        valid=valid_size,
        test=test_size,
        shuffle=shuffle,
        random_seed=random_seed,
    )
    LOGGER.info(
        "Loading datasets: %d training images, %d validation images, %d test images with batch size %d",
        len(train_sampler),
        len(valid_sampler),
        len(test_sampler),
        batch_size,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader
