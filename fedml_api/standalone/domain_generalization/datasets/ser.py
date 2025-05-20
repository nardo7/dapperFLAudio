import numpy as np
import torch
import torch.utils.data as data
import torchaudio
import torchaudio.transforms as audio_transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple, List

from fedml_api.standalone.domain_generalization.datasets.utils.public_dataset import (
    SimpleSpecAugment,
    time_shift_waveform,
)
from fedml_api.standalone.domain_generalization.utils.conf import data_path
from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (
    FederatedDataset,
)
from fedml_api.standalone.domain_generalization.datasets.transforms.denormalization import (
    DeNormalize,
)
from fedml_api.standalone.domain_generalization.backbone.ResNet import (
    resnet10,
    resnet18,
)
from fedml_api.standalone.domain_generalization.backbone.vgg_speech import vgg11
from fedml_api.standalone.domain_generalization.backbone.conv_model import (
    audio_conv_rnn,
)


class AudioDataset(data.Dataset):
    CORRUPTED_FILES = ["1050_MTI_DIS_XX.wav"]

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        data_name=None,
    ) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Build dataset
        self.samples, self.targets, self.samples_by_speakers = self.__build_dataset__()
        # Default audio processing parameters
        self.sample_rate = 16000  # Common sample rate for audio processing
        self.n_mels = 64  # Number of mel bands
        self.n_fft = 1024  # FFT window size
        self.hop_length = 512  # Hop length for STFT

        # Define the mel spectrogram transform
        self.mel_spectrogram = audio_transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        self.spec_augment = SimpleSpecAugment()

        # Define log-mel spectrogram transform for better feature representation
        self.amplitude_to_db = audio_transforms.AmplitudeToDB()

    def __build_dataset__(self):
        """
        Load audio files and emotion labels from the dataset directory
        Returns: list of (audio_path, emotion_label) pairs
        """
        samples: list[str] = []
        targets: list[str] = []
        samples_by_speaker: dict[str, list[tuple[str, int]]] = {}

        if self.data_name == "crema-d":
            # Implementation for CREMA-D dataset
            dataset_path = f"{self.root}/CREMA-D/"
            split_folder = "train" if self.train else "test"
            dataset_path = f"{dataset_path}/{split_folder}"

            # CREMA-D has 6 emotion categories
            # Unified mapping: Neutral (0), Angry (1), Fear (2), Happy (3), Disgust (4), Sad (5)
            emotion_map = {"NEU": 0, "ANG": 1, "FEA": 2, "HAP": 3, "DIS": 4, "SAD": 5}

            import os

            for idx, filename in enumerate(os.listdir(dataset_path)):
                if filename.endswith(".wav"):
                    # Extract emotion from filename (CREMA-D uses specific naming convention)
                    # Format: ActorID_Sentence_Emotion.wav
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        emotion_code = parts[2].split(".")[0]
                        if emotion_code in emotion_map:
                            samples.append(os.path.join(dataset_path, filename))
                            targets.append(emotion_map[emotion_code])

                    # fill speaker's samples
                    speaker = str(int(parts[0]) - 1000)
                    samples_by_speaker[speaker] = (
                        [(filename, idx)]
                        if speaker not in samples_by_speaker
                        else samples_by_speaker[speaker] + [(filename, idx)]
                    )
            self.num_speakers = len(samples_by_speaker)

            # Remove corrupted files
            for corrupted_file in self.CORRUPTED_FILES:
                # print(f"Removing corrupted file: {corrupted_file}")
                for speaker in samples_by_speaker:
                    # speaker_samples_has_corrupted_file = [
                    #     sample[0]
                    #     for sample in samples_by_speaker[speaker]
                    #     if sample[0] == corrupted_file
                    # ]
                    # if len(speaker_samples_has_corrupted_file) > 0:
                    #     print(
                    #         f"1. Speaker {speaker} has corrupted file: {corrupted_file}"
                    #     )

                    samples_by_speaker[speaker] = [
                        sample
                        for sample in samples_by_speaker[speaker]
                        if sample[0] != corrupted_file
                    ]
                    # speaker_samples_has_corrupted_file = [
                    #     sample[0]
                    #     for sample in samples_by_speaker[speaker]
                    #     if sample[0] == corrupted_file
                    # ]
                    # if len(speaker_samples_has_corrupted_file) > 0:
                    #     print(
                    #         f"2. Speaker {speaker} has corrupted file: {corrupted_file}"
                    #     )

        elif self.data_name == "ravdess":
            self.num_speakers = 24
            # Implementation for RAVDESS dataset
            dataset_path = f"{self.root}/RAVDESS/"
            split_folder = "train" if self.train else "test"
            dataset_path = f"{dataset_path}/{split_folder}"

            # RAVDESS emotion encoding (third position in filename):
            # 01=neutral, 02=calm -> map to Neutral (0)
            # 03=happy, 08=surprised -> map to Happy (3)
            # 04=sad -> map to Sad (5)
            # 05=angry -> map to Angry (1)
            # 06=fearful -> map to Fear (2)
            # 07=disgust -> map to Disgust (4)
            unified_emotion_map = {
                "01": 0,  # Neutral
                "02": 0,  # Calm -> Neutral
                "03": 3,  # Happy
                "04": 5,  # Sad
                "05": 1,  # Angry
                "06": 2,  # Fear
                "07": 4,  # Disgust
                "08": 3,  # Surprised -> Happy
            }

            import os

            for filename in os.listdir(dataset_path):
                if filename.endswith(".wav"):
                    # Extract emotion from filename
                    # Format: 03-01-01-01-01-01-01.wav
                    # Position 3 indicates emotion
                    parts = filename.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        if emotion_code in unified_emotion_map:
                            samples.append(os.path.join(dataset_path, filename))
                            targets.append(unified_emotion_map[emotion_code])

        elif self.data_name == "emo-db":
            self.num_speakers = 10
            # Implementation for EMO-DB dataset
            dataset_path = f"{self.root}/EMO-DB/"
            split_folder = "train" if self.train else "test"
            dataset_path = f"{dataset_path}/{split_folder}"

            # EMO-DB emotion mapping
            unified_emotion_map = {
                "N": 0,  # Neutral
                "B": 0,  # Calm -> Neutral
                "F": 3,  # Happy
                "T": 5,  # Sad
                "W": 1,  # Angry
                "A": 2,  # Fear
                "E": 4,  # Disgust
            }

            import os

            for filename in os.listdir(dataset_path):
                if filename.endswith(".wav"):
                    # Extract emotion from filename
                    # Format: XXaYYFa.wav where F is the emotion code
                    emotion_code = filename[5]  # Get the emotion code (6th character)
                    if emotion_code in unified_emotion_map:
                        samples.append(os.path.join(dataset_path, filename))
                        targets.append(unified_emotion_map[emotion_code])

        return samples, targets, samples_by_speaker

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a spectrogram and corresponding label
        """
        audio_path = self.samples[index]
        target = self.targets[index]

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = time_shift_waveform(waveform)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Generate mel spectrogram
        mel_spectrogram = self.mel_spectrogram(waveform)

        # Convert to decibels
        log_mel_spectrogram = self.amplitude_to_db(mel_spectrogram)

        log_mel_spectrogram = self.spec_augment(log_mel_spectrogram)

        # Apply additional transforms if provided
        if self.transform is not None:
            # Convert to PIL Image for compatibility with image transforms
            # spec_min = log_mel_spectrogram.min()
            # spec_max = log_mel_spectrogram.max()
            # spectrogram_image = (
            #     (log_mel_spectrogram - spec_min) / (spec_max - spec_min) * 255.0
            # )
            # spectrogram_image = spectrogram_image.byte().numpy()
            # spectrogram_image = Image.fromarray(spectrogram_image.transpose(1, 2, 0))
            log_mel_spectrogram = self.transform(log_mel_spectrogram)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return log_mel_spectrogram, target

    def __len__(self) -> int:
        return len(self.samples)


class FedLeaSER(FederatedDataset):
    """
    Federated Speech Emotion Recognition Dataset with domain generalization
    """

    NAME = "fl_ser"
    SETTING = "domain_skew"
    DOMAINS_LIST = ["crema-d"]  # , "ravdess", "emo-db"]
    percent_dict = {"crema-d": 0.1}  # , "ravdess": 0.3, "emo-db": 0.8}

    # Number of emotion classes - unified 6 classes
    # Neutral (0), Angry (1), Fear (2), Happy (3), Disgust (4), Sad (5)
    N_CLASS = 6

    # Transform for spectrograms
    Nor_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize spectrogram
            # transforms.RandomCrop(128, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.5), (0.225)),
        ]
    )

    def get_data_loaders(self, selected_domain_list: List[str] = []):
        """
        Get data loaders for the selected domains, or all domains if none specified
        """
        using_list = (
            self.DOMAINS_LIST
            if len(selected_domain_list) == 0
            else selected_domain_list
        )

        nor_transform = self.Nor_TRANSFORM

        train_dataset_list: list[AudioDataset] = []
        test_dataset_list: list[AudioDataset] = []

        test_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),  # Resize spectrogram
                # transforms.ToTensor(),
                self.get_normalization_transform(),
            ]
        )

        # Create datasets for each domain
        for _, domain in enumerate(using_list):
            train_dataset = AudioDataset(
                root=data_path(),
                train=True,
                transform=nor_transform,
                data_name=domain,
            )
            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            test_dataset = AudioDataset(
                root=data_path(),
                train=False,
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
            [transforms.ToPILImage(), FedLeaSER.Nor_TRANSFORM]
        )
        return transform

    @staticmethod
    def get_backbone(parti_num, net_name: str | None):
        """
        Get neural network backbones for the participants
        """
        nets_dict = {
            "resnet10": resnet10,
            "res10": resnet10,
            "res18": resnet18,
            "resnet18": resnet18,
            "VGG11": vgg11,
            "vgg11": vgg11,
            "conv_model": audio_conv_rnn,
        }
        nets_list = []
        if net_name is None:
            for j in range(parti_num):
                nets_list.append(resnet10(FedLeaSER.N_CLASS, input_channels=1))
        else:
            for j in range(parti_num):
                if net_name == "conv_model":
                    # Initialize audio_conv_rnn with proper parameters
                    # feature_size=64 (n_mels), dropout=0.2, label_size=N_CLASS
                    nets_list.append(
                        audio_conv_rnn(
                            feature_size=64, dropout=0.2, label_size=FedLeaSER.N_CLASS
                        )
                    )
                else:
                    nets_list.append(
                        nets_dict[net_name](FedLeaSER.N_CLASS, input_channels=1)
                    )
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5), (0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485), (0.225))
        return transform

    def partition_domain_skew_loaders(
        self,
        train_datasets: list[AudioDataset],
        test_datasets: list[AudioDataset],
    ) -> Tuple[list[DataLoader], list[DataLoader]]:
        """
        Partition datasets into train and test loaders with domain skew
        """
        # ini_len_dict = {}
        # not_used_index_dict = {}
        grouped_by_domain: dict[str, list[AudioDataset]] = {}
        num_workers = 4 if torch.cuda.is_available() else 0

        for domain in self.DOMAINS_LIST:
            # find a dataset with the same domain
            grouped_by_domain[domain] = [
                ds for ds in train_datasets if ds.data_name == domain
            ]

        # count clients per domain
        speaker_ids_per_domain = {}

        for domain in grouped_by_domain:
            speaker_ids_per_domain[domain] = np.random.choice(
                grouped_by_domain[domain][0].num_speakers,
                len(grouped_by_domain[domain]),
                replace=False,
            )

        print("Speaker IDs per domain: ", speaker_ids_per_domain)

        # Initialize dictionaries with dataset indices
        for index in range(len(train_datasets)):
            speaker_idx = speaker_ids_per_domain[domain][index]
            speaker_id = list(train_datasets[index].samples_by_speakers.keys())[
                speaker_idx
            ]
            print(f"Speaker ID chosen for client {index}: {speaker_id}")

            #     name = train_datasets[i].data_name
            #     if name not in not_used_index_dict:
            #         train_dataset = train_datasets[i]
            #         y_train = train_dataset.targets

            #         not_used_index_dict[name] = np.arange(len(y_train))
            #         ini_len_dict[name] = len(y_train)

            # # Create train loaders
            # for index in range(len(train_datasets)):
            # name = train_datasets[index].data_name
            train_dataset = train_datasets[index]

            idxs = [
                sample[1]
                for sample in train_dataset.samples_by_speakers[str(speaker_id)]
            ]

            # percent = self.percent_dict[name]
            selected_idx = np.array(idxs)  # idxs[0 : int(percent * ini_len_dict[name])]

            # not_used_index_dict[name] = idxs[int(percent * ini_len_dict[name]) :]

            train_sampler = SubsetRandomSampler(selected_idx)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.local_batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
            )
            self.train_loaders.append(train_loader)

        # Create test loaders
        for index in range(len(test_datasets)):
            test_dataset = test_datasets[index]

            test_loader = DataLoader(
                test_dataset, batch_size=self.args.local_batch_size, shuffle=False
            )
            self.test_loader.append(test_loader)

        return self.train_loaders, self.test_loader
