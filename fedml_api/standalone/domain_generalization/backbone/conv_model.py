#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import pdb


class conv_classifier(nn.Module):
    def __init__(self, pred, audio_size, txt_size, hidden_size, att=None):
        super(conv_classifier, self).__init__()
        self.dropout_p = 0.2
        self.test_conf = None
        self.rnn_cell = nn.GRU

        num_class = class_dict[pred]

        hidden_size = 64
        self.emo_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 * 2, 128), nn.ReLU(), nn.Linear(128, num_class)
        )

        self.arousal_pred = nn.Sequential(
            nn.Linear(hidden_size * 2 * 2, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.valence_pred = nn.Sequential(
            nn.Linear(hidden_size * 2 * 2, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.audio_conv = nn.Sequential(
            nn.Conv1d(audio_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(self.dropout_p),
        )

        self.text_conv = nn.Sequential(
            nn.Conv1d(txt_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
        )

        self.audio_rnn = self.rnn_cell(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=True,
        )
        self.txt_rnn = self.rnn_cell(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=True,
        )

        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)

    def forward(self, audio, txt_embedding):
        audio, txt_embedding = audio.float(), txt_embedding.float()
        audio = audio.permute(0, 2, 1)
        audio = self.audio_conv(audio)
        audio = audio.permute(0, 2, 1)

        txt_embedding = txt_embedding.permute(0, 2, 1)
        txt_embedding = self.text_conv(txt_embedding)
        txt_embedding = txt_embedding.permute(0, 2, 1)

        audio, h_state = self.audio_rnn(audio)
        txt_embedding, h_state = self.txt_rnn(txt_embedding)

        audio = torch.mean(audio, dim=1)
        txt_embedding = torch.mean(txt_embedding, dim=1)

        final_feat = torch.cat((audio, txt_embedding), 1)
        preds = self.emo_classifier(final_feat)
        arousal = self.arousal_pred(final_feat)
        valence = self.valence_pred(final_feat)
        return preds, arousal, valence


class audio_conv(nn.Module):
    def __init__(self, pred, audio_size, dropout):
        super(audio_conv, self).__init__()
        self.dropout_p = dropout

        self.pred_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))

        self.conv = nn.Sequential(
            nn.Conv1d(audio_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(self.dropout_p),
        )

        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)

    def forward(self, audio):
        audio = audio.float()
        audio = audio.permute(0, 2, 1)
        audio = self.conv(audio)
        audio = audio.permute(0, 2, 1)
        audio = torch.mean(audio, dim=1)

        preds = self.pred_layer(audio)
        preds = torch.log_softmax(preds, dim=1)
        return preds


class audio_conv_rnn(nn.Module):
    def __init__(self, feature_size, dropout, label_size=4, hidden_size=128):
        super(audio_conv_rnn, self).__init__()
        self.name = "audio_conv_rnn"
        self.feature_size = feature_size
        self.dropout_p = dropout

        # Modified classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_size, label_size),
        )

        # Modified conv layers with batch normalization
        self.conv = nn.Sequential(
            nn.Conv1d(128, hidden_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout_p),
        )

        # Modified RNN with more layers
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,  # Increased from 1 to 2
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=True,
        )

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def features(self, x):
        # Extract features from the input
        # Input shape: [batch_size, 3, 128, 128]
        # First convert from image format to audio format
        x = x.mean(
            dim=1, keepdim=True
        )  # Convert from 3 channels to 1, shape: [batch_size, 1, 128, 128]
        x = x.squeeze(1)  # Remove channel dimension, shape: [batch_size, 128, 128]

        # Reshape for conv1d: [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)  # shape: [batch_size, 128, 128]

        # Apply conv and rnn layers
        x = self.conv(x)  # shape: [batch_size, hidden_size, sequence_length]
        x = x.permute(0, 2, 1)  # shape: [batch_size, sequence_length, hidden_size]
        x, _ = self.rnn(x)

        # Return the mean of the sequence
        return torch.mean(x, dim=1)  # shape: [batch_size, hidden_size*2]

    def forward(self, x):
        # Get features
        features = self.features(x)

        # Apply classifier
        return self.classifier(features)


class audio_rnn(nn.Module):
    def __init__(self, feature_size, dropout, label_size=4):
        super(audio_rnn, self).__init__()
        self.dropout_p = dropout
        self.pred_layer = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, label_size)
        )

        self.rnn = nn.GRU(
            input_size=feature_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=True,
        ).cuda()

        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)

    def forward(self, audio, lengths=None):
        if lengths is None:
            # output
            z = torch.mean(audio, dim=1)
        else:
            # rnn module
            audio_packed = pack_padded_sequence(
                audio, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, _ = self.rnn(audio_packed)
            x_output, _ = pad_packed_sequence(
                output_packed, True, total_length=audio.size(1)
            )

        # pooling based on the real sequence length
        z = torch.sum(x_output, dim=1) / torch.unsqueeze(lengths, 1)
        preds = self.pred_layer(z)
        return preds
