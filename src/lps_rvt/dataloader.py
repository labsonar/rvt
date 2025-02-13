""" DataLoader module
"""
import os
import typing
import pandas as pd
import scipy.io as scipy

import lps_rvt.rvt_types as rvt

class BaseLoader:
    """ Base class for loading and filtering test data and artifacts """
    def __init__(self,
                 description_filename: str,
                 artifacts_filename: str,
                 artifact_id_column: str,
                 data_dir: str):
        self.description = pd.read_csv(description_filename)
        self.artifacts = pd.read_csv(artifacts_filename)
        self.artifact_id_column = artifact_id_column
        self.data_dir = data_dir

    def get_files(self,
                  ammunition_types: typing.Optional[typing.List[rvt.Ammunition]] = None,
                  buoys: typing.Optional[typing.List[int]] = None,
                  subsets: typing.Optional[typing.List[rvt.Subset]] = None) -> typing.List[int]:
        """Returns a list of file IDs with the specified restrictions"""
        df_filtered = self.description
        if ammunition_types:
            df_filtered = df_filtered[df_filtered["Type"].isin([ft.value \
                                                            for ft in ammunition_types])]
        if buoys:
            df_filtered = df_filtered[df_filtered["Bouy"].isin(buoys)]
        if subsets:
            df_filtered = df_filtered[df_filtered["Subset"].isin([st.value for st in subsets])]
        return df_filtered["File"].tolist()

    def get_data(self, file_id: typing.Union[int, str], fs: int = 8000):
        """Gets the audio data from a specific file."""
        if fs == 8000:
            rel_filename = f"{file_id}.wav"
        else:
            rel_filename = f"{file_id}-{fs}.wav"
        audio_path = os.path.join(self.data_dir, rel_filename)
        fs, audio_data = scipy.wavfile.read(audio_path)
        return fs, audio_data

    def get_critical_points(self, file_id: typing.Union[int, str], fs: int) -> \
            typing.Tuple[typing.List[int], typing.List[int]]:
        """Returns samples where shots and rebounds occur."""
        expected_detections = []
        expected_rebounds = []
        artifacts_filtered = self.artifacts[self.artifacts[self.artifact_id_column] == file_id]

        for _, artifact in artifacts_filtered.iterrows():
            delta = pd.Timedelta(artifact['Offset']).total_seconds()
            if "Tiro" in artifact["Caracterization"]:
                expected_detections.append(int(delta * fs))
            else:
                expected_rebounds.append(int(delta * fs))

        return expected_detections, expected_rebounds

class DataLoader(BaseLoader):
    """ Loader for test data """
    artifact_id_column = "Test File ID"

    def __init__(self,
                 description_filename: str = "./data/docs/test_files_description.csv",
                 artifacts_filename: str = "./data/docs/test_artifacts.csv"):
        super().__init__(description_filename,
                         artifacts_filename,
                         "Test File ID",
                         "./data/test_files")

class ArtifactLoader(BaseLoader):
    """ Loader for artifact data """
    artifact_id_column = "Artifact File ID"

    def __init__(self,
                 description_filename: str = "./data/docs/test_files_description.csv",
                 artifacts_filename: str = "./data/docs/artifacts_ids.csv"):
        super().__init__(description_filename,
                         artifacts_filename,
                         "Artifact File ID",
                         "./data/artifacts")

    def get_files(self,
                  ammunition_types: typing.Optional[typing.List[rvt.Ammunition]] = None,
                  buoys: typing.Optional[typing.List[int]] = None,
                  subsets: typing.Optional[typing.List[rvt.Subset]] = None) -> typing.List[str]:
        """Returns a list of artifact file IDs with the specified restrictions"""
        files = super().get_files(ammunition_types, buoys, subsets)
        artifacts_df = self.artifacts[self.artifacts["Test File ID"].isin(files)]
        return artifacts_df["Artifact File ID"].tolist()
