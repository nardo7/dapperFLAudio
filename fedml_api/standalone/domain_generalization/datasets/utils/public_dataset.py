from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim


class PublicDataset:
    NAME = None
    SETTING = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> DataLoader:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


def random_loaders(train_dataset: datasets, setting: PublicDataset) -> DataLoader:
    public_scale = setting.args.public_len
    y_train = train_dataset.targets
    n_train = len(y_train)
    idxs = np.random.permutation(n_train)
    if public_scale != None:
        idxs = idxs[0:public_scale]
    train_sampler = SubsetRandomSampler(idxs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=setting.args.public_batch_size,
        sampler=train_sampler,
        num_workers=4,
    )
    setting.train_loader = train_loader

    return setting.train_loader


class PadSequence:
    def __init__(self, max_len: int):
        self.max_len = max_len

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Pads the input tensor to the specified maximum length.
        :param batch: The input tensor to be padded.
        :return: The padded tensor.
        """
        return pad_sequence(audio, self.max_len)


def pad_sequence(audio: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Pads the input tensor to the specified maximum length.
    :param audio: The input tensor to be padded.
    :param max_len: The maximum length to pad to.
    :return: The padded tensor.
    """
    if audio.size(-1) < max_len:
        padding = torch.zeros(1, max_len - audio.size(1))
        # print(padding.shape)
        # print(audio.shape)
        audio = torch.cat((audio, padding), dim=1)
        # print(audio.shape)
    return audio
