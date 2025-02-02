""" DataLoader module
"""
import os
import typing

import pandas as pd
import scipy.io as scipy

import lps_rvt.types as rvt

class DataLoader:
    """ Class to acess and filter the test data"""
    def __init__(self,
                 description_filename: str = "./data/docs/test_files_description.csv",
                 artifacts_filename='data/RVT/test_artifacts.csv'):
        self.description = pd.read_csv(description_filename)
        self.artifacts = pd.read_csv(artifacts_filename)

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

        df_filtered = self.description
        if file_types:
            df_filtered = df_filtered[df_filtered["Type"].isin([ft.value for ft in file_types])]
        if buoys:
            df_filtered = df_filtered[df_filtered["Bouy"].isin(buoys)]
        if subsets:
            df_filtered = df_filtered[df_filtered["Subset"].isin([st.value for st in subsets])]
        return df_filtered["File"].tolist()

    def get_data (self, file_id: int, fs: int = 8000, path='data/RVT/test_files'):
        """Gets the audio data from an specific file.

        Args:
            file_ID (int): file identification.
            frequency (int): the audio's sampling frequency.
            path (str): where to find the file.

        Returns:
            fs (int): sampling_rate.
            audio_data (ndarray): audio data.

        """
        if fs == 8000:
            rel_filename = f"{file_id}.wav"
        else:
            rel_filename = f"{file_id}-{fs}.wav"

        audio_path = os.path.join(path, rel_filename)
        fs, audio_data = scipy.wavfile.read(audio_path)

        return fs, audio_data

    def get_excepted_detections(self, file_id: int, fs: int):

        expected = []
        for offset in self.artifacts[self.artifacts["Test File ID"] == file_id]["Offset"]:
            delta = pd.Timedelta(offset).total_seconds()
            expected.append(int(delta * fs))

        return expected
