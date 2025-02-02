""" DataLoader module
"""
import os
import typing

import pandas as pd
import scipy.io as scipy

import lps_rvt.types as rvt

class DataLoader:
    """ Class to acess and filter the test data"""
    def __init__(self, csv_path: str = "./data/docs/test_files_description.csv"):
        self.data = pd.read_csv(csv_path)

    def get_files(self,
                  file_types: typing.Optional[typing.List[rvt.Ammunition]] = None,
                  buoys: typing.Optional[typing.List[int]] = None,
                  subsets: typing.Optional[typing.List[rvt.Subset]] = None) -> typing.List[int]:
        """Returns a list of file IDs with the specified restrictions

        Args:
            file_types (typing.Optional[typing.List[rvt.DataType]], optional):
                List of type of ammunition. Defaults to all.
            buoys (typing.Optional[typing.List[int]], optional):
                List of bouys. Defaults to all.
            subsets (typing.Optional[typing.List[rvt.SubsetType]], optional):
                List of subsets. Defaults to all.

        Returns:
            typing.List[int]: list of file IDs with the specified restrictions
        """

        df_filtered = self.data
        if file_types:
            df_filtered = df_filtered[df_filtered["Type"].isin([ft.value for ft in file_types])]
        if buoys:
            df_filtered = df_filtered[df_filtered["Bouy"].isin(buoys)]
        if subsets:
            df_filtered = df_filtered[df_filtered["Subset"].isin([st.value for st in subsets])]
        return df_filtered["File"].tolist()

    def get_data (self, file_id: int, frequency: int = 8000, path='data/RVT/test_files'):
        """Gets the audio data from an specific file.

        Args:
            file_ID (int): file identification.
            frequency (int): the audio's sampling frequency.
            path (str): where to find the file.

        Returns:
            fs (int): sampling_rate.
            audio_data (ndarray): audio data.

        """
        if frequency == 8000:
            rel_filename = f"{file_id}.wav"
        else:
            rel_filename = f"{file_id}-{frequency}.wav"

        audio_path = os.path.join(path, rel_filename)
        fs, audio_data = scipy.wavfile.read(audio_path)

        return fs, audio_data
