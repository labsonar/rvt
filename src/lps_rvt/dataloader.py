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
    """ Loader for new marambaia data files."""
    artifact_id_column = "wav filename"

    def __init__(self,
                 description_filename: str = "./data/new_data/consolidated_marambaia_logs.csv",
                 artifacts_filename: str = "./data/new_data/consolidated_marambaia_logs.csv",
                 data_dir: str = "./data/new_data"):
        # Using the same CSV for both description and artifacts
        self.description = pd.read_csv(description_filename, sep=';')
        self.artifacts = pd.read_csv(artifacts_filename, sep=';')
        self.artifact_id_column = "wav filename"
        self.data_dir = data_dir

        # Pre-process columns for filtering
        # Map 'splash valido' to Ammunition Type
        def map_ammunition(val):
            val = str(val).strip()
            if val.lower() == 'exsup':
                return rvt.Ammunition.EXSUP.value
            elif 'gae' in val.lower():
                return rvt.Ammunition.GAE.value
            elif 'he3m' in val.lower():
                return rvt.Ammunition.HE3M.value
            return None
        
        self.description['Type'] = self.description['splash valido'].apply(map_ammunition)
        
        # Map 'boia' to integer ID
        def map_buoy(val):
            val = str(val).strip()
            try:
                if val.lower().startswith('boia'):
                    part = val.lower().split('_')[0]
                    return int(''.join(filter(str.isdigit, part)))
            except:
                pass
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
        
        # TODO: Ignoring subsets for now as the CSV doesn't have them
        
        unique_files = df_filtered["wav filename"].unique()
        return [f.replace(".wav", "") for f in unique_files]

    def get_data(self, file_id: typing.Union[int, str], fs: int = 8000):
        """Gets the audio data from a specific file."""
        
        rel_filename = f"{file_id}.wav"
        audio_path = os.path.join(self.data_dir, rel_filename)
        fs, audio_data = scipy.wavfile.read(audio_path)
        return fs, audio_data

    def get_critical_points(self, file_id: typing.Union[int, str], fs: int) -> \
            typing.Tuple[typing.List[int], typing.List[int]]:
        """Returns samples where shots occur based on 'splash moment'."""
        expected_detections = []
        expected_rebounds = []
        
        # Filter artifacts for this file
        filename = f"{file_id}.wav"
        artifacts_filtered = self.artifacts[self.artifacts[self.artifact_id_column] == filename]

        for _, artifact in artifacts_filtered.iterrows():
            # Parse 'splash moment' M:SS.mmm
            moment_str = artifact['splash moment']
            try:
                if str(moment_str).count(':') == 1:
                    match_time = "00:" + str(moment_str)
                else:
                    match_time = str(moment_str)
                
                delta = pd.Timedelta(match_time).total_seconds()
                
                # Check validation column
                if str(artifact.get('splash valido', '')).strip().lower() == 'exsup':
                     expected_detections.append(int(delta * fs))
                elif str(artifact.get('splash valido', '')).strip() == '1_GAE':
                     expected_detections.append(int(delta * fs))
                else:
                    expected_detections.append(int(delta * fs))

            except Exception as e:
                print(f"Error parsing time for {filename}: {e}")
                continue

        return expected_detections, expected_rebounds
