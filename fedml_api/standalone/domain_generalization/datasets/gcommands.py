import os
import sys
import torch
import torchaudio
from torchvision.transforms import Compose
from torchaudio.datasets import SPEECHCOMMANDS

sys.path.append("./")
conf_path = os.getcwd()
print(conf_path)

from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (  # noqa: E402
    FederatedDataset,
)
from fedml_api.standalone.domain_generalization.utils.conf import data_path  # noqa: E402


class GCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, root, download=False, transform=None, label_transform=None):
        super().__init__(root, download=download)
        self.data_name = "gcommands"
        self.root = root
        self.transform = None
        self.label_transform = label_transform

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = (
            super().__getitem__(index)
        )
        if self.transform is not None:
            waveform = self.transform(waveform)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return waveform, label


class FederatedGCommandsDataset(FederatedDataset):
    NAME = "gcommands"
    DOMAIN_LIST = ["gcommands"]
    percent_dict = {
        "gcommands": 0.2,
    }

    transform = Compose(
        [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=128,
                n_fft=1024,
                win_length=1024,
                hop_length=160,
                window_fn=torch.hann_window,
            ),
            torchaudio.transforms.AmplitudeToDB(),
        ]
    )

    def get_data_loaders(self, selected_domain_list=...):
        using_list = (
            self.DOMAINS_LIST
            if len(selected_domain_list) == 0
            else selected_domain_list
        )

        for domain in using_list:
            if domain not in self.percent_dict.keys():
                raise ValueError(
                    f"Domain {domain} not in the dataset {self.NAME} domain list"
                )

            if domain == "gcommands":
                dataset = SPEECHCOMMANDS(root=data_path(), download=True)


if __name__ == "__main__":
    dataset = GCommandsDataset(root=data_path(), download=True)
    print(dataset[0])
