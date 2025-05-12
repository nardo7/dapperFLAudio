import sys

import torch

sys.path.append("../../../..")

from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (
    FederatedDataset,
)
import torchaudio.datasets as datasets
from torch.utils.data import DataLoader
import torchaudio
from fedml_api.standalone.domain_generalization.utils.conf import data_path


class CremaD(FederatedDataset):
    NAME = "cremad"
    DOMAIN_LIST = ["cremad"]
    percent_dict = {
        "cremad": 0.2,
    }

    def get_data_loaders(self, selected_domain_list=[]):
        pass
        # dataset = datasets.SPEECHCOMMANDS()


def mel_spectrogram(audio_file_path, n_fft=1024, feature_len=128):
    window_size = n_fft
    window_hop = 160
    n_mels = feature_len
    window_fn = torch.hann_window

    audio, sample_rate = torchaudio.load(audio_file_path)
    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = transform_model(audio)

    audio_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=int(window_size),
        hop_length=int(window_hop),
        window_fn=window_fn,
    )
    # print(audio_file_path)
    audio_amp_to_db = torchaudio.transforms.AmplitudeToDB()
    return audio_amp_to_db(audio_transform(audio).detach())[0].cpu().numpy().T


if __name__ == "__main__":
    ds = datasets.iemocap.IEMOCAP()

    print(ds[0])
