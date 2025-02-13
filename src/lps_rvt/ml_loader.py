"""Module to load the detections and background as pytorch datasets
"""
import typing
import numpy as np
import pandas as pd
import scipy.io as scipy

import torch
import torch.utils.data as torch_data

import lps_rvt.rvt_types as rvt

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
        self.transform = transform

        if subset is not None:
            self.data = self.data[self.data['Subset'] == str(self.subset)]

        print(subset, ": ", len(self.data[self.data['Classification'] == 1]), "/", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        wav_file = sample['outputfile']
        classification = sample['Classificação']

        _, audio_data = scipy.wavfile.read(wav_file)

        audio_data = audio_data / np.max(np.abs(audio_data), axis=0)

        if self.transform:
            audio_data = self.transform(audio_data)

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        classification_tensor = torch.tensor(classification, dtype=torch.long)

        return audio_tensor, classification_tensor

    @staticmethod
    def get_dataloaders(batch_size=32) -> typing.Dict[rvt.Subset, 'AudioDataset']:
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
            dataset = AudioDataset(subset=subset)
            dataloaders[subset] = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloaders

