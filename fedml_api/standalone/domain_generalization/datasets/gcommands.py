import os
import sys
import numpy as np
import torch
import torchaudio
from torchvision.transforms import Compose, Normalize, Resize
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler


sys.path.append("./")
conf_path = os.getcwd()
from fedml_api.standalone.domain_generalization.backbone.vgg_speech import vgg11

from fedml_api.standalone.domain_generalization.backbone.ResNet import resnet10
from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (  # noqa: E402
    FederatedDataset,
)
from fedml_api.standalone.domain_generalization.utils.conf import data_path  # noqa: E402
from fedml_api.standalone.domain_generalization.datasets.utils.public_dataset import (
    PadSequence,
)  # noqa: E402


class FromLabelsToIdx:
    def __init__(self, labels: list[str]):
        self.labels = labels

    def __call__(self, label):
        return torch.tensor(self.labels.index(label))


class Permute:
    def __init__(self, permute: list[int]):
        self.permute = permute

    def __call__(self, x: torch.Tensor):
        return x.permute(self.permute)


class GCommandsDataset(SPEECHCOMMANDS):
    def __init__(
        self,
        root,
        download=False,
        transform=None,
        label_transform=None,
        subset: str = None,
        preload: bool = False,
    ):
        super().__init__(root, download=download, subset=subset)
        self.data_name = "gcommands"
        self.root = root
        self.transform = transform
        self.label_transform = label_transform
        subfolders = os.listdir(self._path)
        self.classes = [
            subfolder
            for subfolder in subfolders
            if os.path.isdir(os.path.join(self._path, subfolder))
        ]
        self.classes = [
            label for label in self.classes if label != "_background_noise_"
        ]

        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None

        if preload:
            self._preload_data()

    def _preload_data(self):
        # Preload the data into memory
        if os.path.exists(os.path.join(self._path, "preload.pt")):
            print("Preloaded data already exists. Loading from file.")
            data = torch.load(os.path.join(self._path, "preload.pt"))
            self.data = data["data"]
            self.targets = data["targets"]
            return

        self.data: torch.Tensor = None
        targets: np.ndarray = np.zeros(self.__len__())
        for i in tqdm(
            range(self.__len__()),
            desc="Preloading data",
            total=self.__len__(),
            unit="file",
        ):
            waveform, sample_rate, label, speaker_id, utterance_number = (
                super().__getitem__(i)
            )
            if self.transform is not None:
                waveform = self.transform(waveform)
            if self.label_transform is not None:
                label = self.label_transform(label)
            if self.data is None:
                self.data = torch.empty((len(self), *waveform.shape))
            self.data[i] = waveform
            targets[i] = label
        self.targets = torch.tensor(targets, dtype=torch.int16)
        path_to_store = os.path.join(self._path, "preload.pt")
        torch.save(
            {
                "data": self.data,
                "targets": self.targets,
            },
            path_to_store,
        )

    def __getitem__(self, index):
        if self.data is not None:
            return (
                self.data[index].to(torch.float32),
                self.targets[index].to(torch.float32),
            )

        waveform, sample_rate, label, speaker_id, utterance_number = (
            super().__getitem__(index)
        )
        if self.transform is not None:
            waveform = self.transform(waveform)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return waveform, label

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.classes.index(word))

    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.classes[index]

    def set_label_transform(self, label_transform):
        self.label_transform = label_transform

    def set_transform(self, transform):
        self.transform = transform


class FederatedGCommandsDataset(FederatedDataset):
    NAME = "gcommands"
    DOMAINS_LIST = ["gcommands"]
    percent_dict = {
        "gcommands": 0.01,
    }

    N_CLASS = 35

    transform = Compose(
        [
            PadSequence(16000),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=128,
                n_fft=1024,
                win_length=1024,
                hop_length=160,
                window_fn=torch.hann_window,
            ),
            Resize((64, 64)),
            torchaudio.transforms.AmplitudeToDB(),
            # normalize to the mean and std of the dataset and get the
            # results:
            #  (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-5)
            Normalize(mean=0.5, std=0.5),
            Permute((0, 2, 1)),
        ]
    )

    @staticmethod
    def get_backbone(parti_num, names_list):
        nets_dict = {
            "resnet10": resnet10,
            "res10": resnet10,
            "VGG11": vgg11,
            "vgg11": vgg11,
            # "resnet12": resnet12,
            # "res18": resnet18,
            # "resnet18": resnet18,
            # "efficient": EfficientNetB0,
            # "mobilnet": MobileNetV2,
        }

        nets_list = []
        if names_list is None:
            for j in range(parti_num):
                nets_list.append(
                    resnet10(FederatedGCommandsDataset.N_CLASS, input_channels=1)
                )
        else:
            for j in range(parti_num):
                # net_name = names_list[j]
                net_name = names_list
                nets_list.append(
                    nets_dict[net_name](
                        FederatedGCommandsDataset.N_CLASS, input_channels=1
                    )
                )
        return nets_list

    def get_data_loaders(self, selected_domain_list=[]):
        using_list = (
            self.DOMAINS_LIST
            if len(selected_domain_list) == 0
            else selected_domain_list
        )

        train_ds_list = []
        test_ds_list = []

        for domain in using_list:
            if domain not in self.percent_dict.keys():
                raise ValueError(
                    f"Domain {domain} not in the dataset {self.NAME} domain list"
                )

            if domain == "gcommands":
                dataset = GCommandsDataset(
                    root=data_path(),
                    download=True,
                    transform=self.transform,
                    subset="training",
                )

                dataset.set_label_transform(FromLabelsToIdx(dataset.classes))
                dataset._preload_data()
                train_ds_list.append(dataset)

        for domain in self.DOMAINS_LIST:
            if domain not in self.percent_dict.keys():
                raise ValueError(
                    f"Domain {domain} not in the dataset {self.NAME} domain list"
                )

            if domain == "gcommands":
                dataset = GCommandsDataset(
                    root=data_path(),
                    download=True,
                    transform=self.transform,
                    subset="testing",
                )

                dataset.set_label_transform(FromLabelsToIdx(dataset.classes))
                test_ds_list.append(dataset)

        trainlds, testlds = self.partition_domain_skew_loaders(
            train_ds_list,
            test_ds_list,
        )

        return trainlds, testlds

    def partition_domain_skew_loaders(
        self,
        train_ds_list: list[GCommandsDataset],
        test_ds_list: list[GCommandsDataset],
    ):
        # Partition the dataset into train and test loaders
        train_loaders = []
        test_loaders = []

        for train_ds in train_ds_list:
            idxs = np.random.choice(
                np.arange(len(train_ds)),
                int(len(train_ds) * self.percent_dict["gcommands"]),
                replace=False,
            )
            train_sampler = SubsetRandomSampler(idxs)
            train_loader = DataLoader(
                train_ds,
                batch_size=self.args.local_batch_size,
                sampler=train_sampler,
            )
            train_loaders.append(train_loader)

        for test_ds in test_ds_list:
            test_loader = DataLoader(test_ds, batch_size=self.args.local_batch_size)
            test_loaders.append(test_loader)

        return train_loaders, test_loaders


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    transform = None
    transform = Compose(
        [
            PadSequence(16000),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=128,
                n_fft=1024,
                win_length=1024,
                hop_length=160,
                window_fn=torch.hann_window,
            ),
            torchaudio.transforms.AmplitudeToDB(),
            Resize((64, 64)),
            Normalize(mean=0.5, std=0.5),
            Permute((0, 2, 1)),
        ]
    )

    dataset = GCommandsDataset(root=data_path(), download=True, subset="training")
    dataset.set_transform(transform)
    dataset.set_label_transform(FromLabelsToIdx(dataset.classes))
    dataset._preload_data()
    print(len(dataset.classes))
    # for i in range(50):
    #     waveform, label = dataset[i]
    #     print(waveform.shape)
    #     print(label)

    # plt.figure()
    waveform, label = dataset[47]
    print(waveform.shape)
    print(dataset._path)
    print(len(dataset))
    plt.imshow(waveform[0].numpy(), cmap="gray")
    plt.title(label)
    plt.show()
