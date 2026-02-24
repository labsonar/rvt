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

class MarambaiaLoader(BaseLoader):
    """ Loader for new marambaia data format """
    artifact_id_column = "arquivo"

    def __init__(self,
                 description_filename: str = "./data/marambaia/marambaia_artifacts.csv",
                 artifacts_filename: str = "./data/marambaia/marambaia_artifacts.csv",
                 data_dir: str = "./data/marambaia/new_data",
                 ignore_ricochets: bool = True,
                 ignore_failures: bool = True):
        # Using the same CSV for both description and artifacts
        self.description = pd.read_csv(description_filename, sep=',')
        self.artifacts = pd.read_csv(artifacts_filename, sep=',')
        self.artifact_id_column = "arquivo"
        self.data_dir = data_dir
        self.ignore_ricochets = ignore_ricochets
        self.ignore_failures = ignore_failures

        # Pre-process columns for filtering
        def map_ammunition(val):
            val = str(val).strip()
            if val.lower() == 'exsup':
                return rvt.Ammunition.EXSUP.value
            elif 'gae' in val.lower():
                return rvt.Ammunition.GAE.value
            elif 'he3m' in val.lower():
                return rvt.Ammunition.HE3M.value
            return None
        
        self.description['Type'] = self.description['ammo'].apply(map_ammunition)
        
        def map_buoy(val):
            try:
                return int(val)
            except:
                return -1

        self.description['Bouy'] = self.description['boia'].apply(map_buoy)

    
    def get_files(self,
                  ammunition_types: typing.Optional[typing.List[rvt.Ammunition]] = None,
                  buoys: typing.Optional[typing.List[int]] = None,
                  subsets: typing.Optional[typing.List[rvt.Subset]] = None) -> typing.List[str]:
        """Returns a list of all file IDs (filenames without extension)"""
        
        df_filtered = self.description

        if ammunition_types:
            df_filtered = df_filtered[df_filtered["Type"].isin([ft.value \
                                                            for ft in ammunition_types])]
        if buoys:
            df_filtered = df_filtered[df_filtered["Bouy"].isin(buoys)]
        
        # TODO: Ignoring subsets for now
        unique_files = df_filtered[self.artifact_id_column].unique()
        return [f.replace(".wav", "") for f in unique_files]

    def get_data(self, file_id: typing.Union[int, str], fs: int = 8000):
        """Gets the audio data from a specific file."""
        
        rel_filename = f"{file_id}.wav"
        audio_path = os.path.join(self.data_dir, rel_filename)
        fs, audio_data = scipy.wavfile.read(audio_path)
        return fs, audio_data

    def get_critical_points(self, file_id: typing.Union[int, str], fs: int) -> \
            typing.Tuple[typing.List[int], typing.List[int]]:
        """Returns samples where shots occur based on 't_splash'."""
        expected_detections = []
        expected_rebounds = []
        
        # Filter artifacts for this file. 
        artifacts_filtered = self.artifacts[self.artifacts[self.artifact_id_column] == str(file_id)]

        for _, artifact in artifacts_filtered.iterrows():
            # Parse 't_splash' M:SS.mmm
            moment_str = artifact['t_splash']
            if pd.isna(moment_str):
                continue
                
            try:
                if str(moment_str).count(':') == 1:
                    match_time = "00:" + str(moment_str)
                else:
                    match_time = str(moment_str)
                
                delta = pd.Timedelta(match_time).total_seconds()

                is_ricochet = str(artifact.get('ricochete', '')).strip().lower() == 'sim'
                is_failure = str(artifact.get('falha', '')).strip().lower() == 'sim'

                if (self.ignore_ricochets and is_ricochet) or (self.ignore_failures and is_failure):
                    expected_rebounds.append(int(delta * fs))
                else:
                    expected_detections.append(int(delta * fs))

            except Exception as e:
                print(f"Error parsing time for {file_id}: {e}")
                continue

        return expected_detections, expected_rebounds
