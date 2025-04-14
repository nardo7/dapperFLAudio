import numpy as np
import torchvision.transforms as transforms
from fedml_api.standalone.domain_generalization.utils.conf import data_path
from PIL import Image
from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (
    FederatedDataset,
)
import torch.utils.data as data
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
from fedml_api.standalone.domain_generalization.datasets.transforms.denormalization import (
    DeNormalize,
)
from fedml_api.standalone.domain_generalization.backbone.ResNet import (
    resnet10,
    resnet12,
    resnet18,
)
from fedml_api.standalone.domain_generalization.backbone.efficientnet import (
    EfficientNetB0,
)
from fedml_api.standalone.domain_generalization.backbone.mobilnet_v2 import MobileNetV2
from torchvision.datasets import MNIST, SVHN, ImageFolder, DatasetFolder, USPS

from typing import Union


class MyDigits(data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        data_name=None,
    ) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.data_name == "mnist":
            dataobj = MNIST(
                self.root,
                self.train,
                self.transform,
                self.target_transform,
                self.download,
            )
            self.targets = dataobj.targets
        elif self.data_name == "usps":
            dataobj = USPS(
                self.root,
                self.train,
                self.transform,
                self.target_transform,
                self.download,
            )
            self.targets = dataobj.targets
        elif self.data_name == "svhn":
            if self.train:
                dataobj = SVHN(
                    self.root,
                    "train",
                    self.transform,
                    self.target_transform,
                    self.download,
                )
            else:
                dataobj = SVHN(
                    self.root,
                    "test",
                    self.transform,
                    self.target_transform,
                    self.download,
                )
            self.targets = dataobj.labels
        return dataobj

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.dataset[index]
        # img = Image.fromarray(img, mode="RGB")

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.dataset)


class ImageFolder_Custom(DatasetFolder):
    def __init__(
        self, data_name, root, train=True, transform=None, target_transform=None
    ):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(
                self.root + self.data_name + "/train/",
                self.transform,
                self.target_transform,
            )
        else:
            self.imagefolder_obj = ImageFolder(
                self.root + self.data_name + "/val/",
                self.transform,
                self.target_transform,
            )
        self.targets = self.imagefolder_obj.targets

    def __getitem__(self, index):
        img, target = self.imagefolder_obj[index]
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imagefolder_obj)


Dataset = Union[MyDigits, ImageFolder_Custom]


class FedLeaDigits(FederatedDataset):
    NAME = "fl_digits"
    SETTING = "domain_skew"
    DOMAINS_LIST = ["mnist", "usps", "svhn", "syn"]
    percent_dict = {"mnist": 0.01, "usps": 0.01, "svhn": 0.01, "syn": 0.01}

    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    Nor_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    Singel_Channel_Nor_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    def get_data_loaders(self, selected_domain_list=[]):
        using_list = (
            self.DOMAINS_LIST
            if len(selected_domain_list) == 0
            else selected_domain_list
        )

        nor_transform = self.Nor_TRANSFORM
        sin_chan_nor_transform = self.Singel_Channel_Nor_TRANSFORM

        train_dataset_list: list[Dataset] = []
        test_dataset_list: list[Dataset] = []

        test_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                self.get_normalization_transform(),
            ]
        )

        sin_chan_test_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                self.get_normalization_transform(),
            ]
        )

        for _, domain in enumerate(using_list):
            if domain == "syn":
                train_dataset = ImageFolder_Custom(
                    data_name=domain,
                    root=data_path(),
                    train=True,
                    transform=nor_transform,
                )
            else:
                # if domain in ['mnist', 'usps']:
                #     train_dataset = MyDigits(data_path(), train=True,
                #                              download=True, transform=sin_chan_nor_transform, data_name=domain)
                if domain == "mnist":
                    train_dataset = MyDigits(
                        data_path(),
                        train=True,
                        download=True,
                        transform=sin_chan_nor_transform,
                        data_name=domain,
                    )
                elif domain == "usps":
                    train_dataset = MyDigits(
                        data_path() + "/USPS",
                        train=True,
                        download=True,
                        transform=sin_chan_nor_transform,
                        data_name=domain,
                    )
                else:
                    # train_dataset = MyDigits(data_path(), train=True,
                    #                          download=True, transform=nor_transform, data_name=domain)
                    train_dataset = MyDigits(
                        data_path() + "/SVHN",
                        train=True,
                        download=True,
                        transform=nor_transform,
                        data_name=domain,
                    )
            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            if domain == "syn":
                test_dataset = ImageFolder_Custom(
                    data_name=domain,
                    root=data_path(),
                    train=False,
                    transform=test_transform,
                )
            else:
                # if domain in ['mnist', 'usps']:
                #     test_dataset = MyDigits(data_path(), train=False,
                #                             download=True, transform=sin_chan_test_transform, data_name=domain)
                if domain == "mnist":
                    test_dataset = MyDigits(
                        data_path(),
                        train=False,
                        download=True,
                        transform=sin_chan_test_transform,
                        data_name=domain,
                    )
                elif domain == "usps":
                    test_dataset = MyDigits(
                        data_path() + "/USPS",
                        train=False,
                        download=True,
                        transform=sin_chan_test_transform,
                        data_name=domain,
                    )
                else:
                    # test_dataset = MyDigits(data_path(), train=False,
                    #                         download=True, transform=test_transform, data_name=domain)
                    test_dataset = MyDigits(
                        data_path() + "/SVHN",
                        train=False,
                        download=True,
                        transform=test_transform,
                        data_name=domain,
                    )

            test_dataset_list.append(test_dataset)
        traindls, testdls = self.partition_domain_skew_loaders(
            train_dataset_list, test_dataset_list
        )

        return traindls, testdls

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaDigits.Nor_TRANSFORM]
        )
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list):
        nets_dict = {
            "resnet10": resnet10,
            "res10": resnet10,
            "resnet12": resnet12,
            "res18": resnet18,
            "resnet18": resnet18,
            "efficient": EfficientNetB0,
            "mobilnet": MobileNetV2,
        }
        nets_list = []
        if names_list is None:
            for j in range(parti_num):
                nets_list.append(resnet10(FedLeaDigits.N_CLASS))
        else:
            for j in range(parti_num):
                # net_name = names_list[j]
                net_name = names_list
                nets_list.append(nets_dict[net_name](FedLeaDigits.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return transform

    def partition_domain_skew_loaders(
        self,
        train_datasets: list[Dataset],
        test_datasets: list[Dataset],
    ) -> Tuple[list[DataLoader], list[DataLoader]]:
        ini_len_dict = {}
        not_used_index_dict = {}
        for i in range(len(train_datasets)):
            name = train_datasets[i].data_name
            if name not in not_used_index_dict:
                if name == "svhn":
                    train_dataset = train_datasets[i]
                    y_train = train_dataset.targets
                elif name == "syn":
                    train_dataset = train_datasets[i]
                    y_train = train_dataset.targets
                else:
                    train_dataset = train_datasets[i]
                    y_train = train_dataset.targets

                not_used_index_dict[name] = np.arange(len(y_train))
                ini_len_dict[name] = len(y_train)

        for index in range(len(train_datasets)):
            name = train_datasets[index].data_name

            if name == "syn":
                train_dataset = train_datasets[index]
            else:
                train_dataset = train_datasets[index]

            idxs = np.random.permutation(not_used_index_dict[name])

            percent = self.percent_dict[name]
            selected_idx = idxs[0 : int(percent * ini_len_dict[name])]

            not_used_index_dict[name] = idxs[int(percent * ini_len_dict[name]) :]

            train_sampler = SubsetRandomSampler(selected_idx)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.local_batch_size,
                sampler=train_sampler,
            )
            self.train_loaders.append(train_loader)

        for index in range(len(test_datasets)):
            name = test_datasets[index].data_name
            if name == "syn":
                test_dataset = test_datasets[index]
            else:
                test_dataset = test_datasets[index]

            test_loader = DataLoader(
                test_dataset, batch_size=self.args.local_batch_size, shuffle=False
            )
            self.test_loader.append(test_loader)

        return self.train_loaders, self.test_loader
