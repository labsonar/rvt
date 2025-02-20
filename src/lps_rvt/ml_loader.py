"""Module to load the detections and background as pytorch datasets
run test/prepare_dataset.py to extract the necessary audios
"""
import typing
import numpy as np
import pandas as pd
import scipy.io as scipy

import torch
import torch.utils.data as torch_data
import torchaudio

import lps_rvt.rvt_types as rvt
import lps_sp.signal as lps_sp


class SpectrogramTransform(torch.nn.Module):
    def __init__(self, n_fft=512, hop_length=128, power=2):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power)

    def forward(self, waveform):
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        return self.spectrogram(waveform)

class MelSpectrogramTransform(torch.nn.Module):
    def __init__(self, n_fft=512, hop_length=128, n_mels=64, power=2):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)

    def forward(self, waveform):
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        return self.mel_spectrogram(waveform)

class AudioDataset(torch_data.Dataset):
    """Class to represent"""

    def __init__(self, subset: rvt.Subset=None, transform=None, csv_file: str = "./data/docs/ml_info.csv"):
        """
        Args:
            subset (string, optional): Selected subset, see rvt_types
            transform (callable, optional): transform function to be apllied to data.
            csv_file (string): path to ml_info.csv.
        """
        self.data = pd.read_csv(csv_file)
        self.subset = subset
        self.transform = transform if transform is not None else lps_sp.Normalization.MIN_MAX_ZERO_CENTERED

        if subset is not None:
            self.data = self.data[self.data['Subset'] == str(self.subset)]

        # print(subset, ": ", len(self.data[self.data['Classification'] == 1]), "/", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        wav_file = sample['outputfile']
        classification = sample['Classification']

        _, audio_data = scipy.wavfile.read(wav_file)

        audio_data = audio_data / np.max(np.abs(audio_data), axis=0)

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        if self.transform:
            audio_tensor = self.transform(audio_tensor)

        classification_tensor = torch.tensor(classification, dtype=torch.long)

        return audio_tensor, classification_tensor

    @staticmethod
    def get_dataloaders(batch_size=32, transform=None) -> typing.Dict[rvt.Subset, 'AudioDataset']:
        """
        Cria os dataloaders para os três subsets: 'train', 'val', 'test'.

        Args:
            csv_file (str): Caminho para o arquivo CSV com informações das amostras.
            batch_size (int): Tamanho do batch para o DataLoader.

        Returns:
            dict: Dicionário contendo os DataLoaders para 'train', 'val', 'test'.
        """
        dataloaders = {}

        for subset in rvt.Subset:
            dataset = AudioDataset(subset=subset, transform=transform)
            dataloaders[subset] = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloaders

