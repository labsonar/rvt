import typing
import os
from scipy.io import wavfile
import numpy as np
import pandas as pd

import lps_utils.utils as lps_utils
from lps_sp.signal import decimate

class DataLoader():

    def __init__(self):
        self.data = pd.read_csv('data/docs/test_files_description.csv')
        self.ammo_dict = self.dictByAmmo()
        self.bouy_dict = self.dictByBouy()
        self.test_files_dict = self.dictByAudioType()
    
    
    def dictByAmmo (self) -> \
        typing.Dict[str, typing.List[int]]:
        """Generates a dictionary containing the file IDs to the data according to the ammunition.

        Returns:
            ammo_dict (dict): dictionary containing the IDs.
        """

        ammo_dict = {
            'EX-SUP': [],
            'HE3m': [],
            'GAE': [],
        }

        for _, row in self.data.iterrows():
            # row -> pd.Series
            ammo_type = row['Type']
            ammo_dict[ammo_type].append(row['File'])
        
        return ammo_dict


    def dictByBouy (self) -> \
        typing.Dict[int, typing.List[int]]:
        """Generates a dictionary containing the file IDs to the data according to the bouy.

        Returns:
            bouy_dict (dict): dictionary containing the IDs.
        """

        bouy_dict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: []
        }

        for _, row in self.data.iterrows():
            # row -> pd.Series
            bouy_ID = row['Bouy']
            bouy_dict[bouy_ID].append(row['File'])
        
        return bouy_dict
    

    def dictByAudioType (self) -> \
        typing.Dict[int, typing.List[int]]:
        """Generates a dictionary containing the file IDs to the data according to the file 
            category, test file or not.

        Returns:
            audio_dict (dict): dictionary containing the IDs.
        """

        audio_dict = {
            1: [],
            0: []
        }

        for _, row in self.data.iterrows():
            # row -> pd.Series
            audio_type = row['Test']
            audio_dict[audio_type].append(row['File'])
        
        return audio_dict  
        

    def getID (self, ammo_type: str = None, bouy_ID: int = None, test_files: bool = False):
        """Gets the file ID according to the applied restrictions.

        Args:
            ammo_type (str): the type of ammunition related to the audio.
            bouy_ID (int): the bouy used to record the audio.
            test_files (bool): whether or not it's a test file.

        Returns:
            intersec (set): a set of the file IDs which correspond the restrictions.        
        """

        if ammo_type and bouy_ID and test_files:
            intersec = set(self.ammo_dict[ammo_type]) & set(self.bouy_dict[bouy_ID]) & \
                        set(self.test_files_dict[test_files])
                  
        elif ammo_type and bouy_ID:
            intersec = set(self.ammo_dict[ammo_type]) & set(self.bouy_dict[bouy_ID])

        elif ammo_type and test_files:
            intersec = set(self.ammo_dict[ammo_type]) & set(self.test_files_dict[test_files])

        elif bouy_ID and test_files:
            intersec = set(self.bouy_dict[bouy_ID]) & set(self.test_files_dict[test_files])
        
        elif ammo_type:
            intersec = set(self.ammo_dict[ammo_type])

        elif bouy_ID :
            intersec = set(self.bouy_dict[bouy_ID])

        elif test_files:
            intersec = set(self.test_files_dict[test_files])
        
        else:
            print('No restrictions specified.')
            return set()
            
        if not intersec:
            print('\nNo files match the applied restrictions.\n')
            
        return intersec
    

    def getData (self, file_ID: int, frequency: int = 8000, path='data/RVT/test_files'):
        """Gets the audio data from an specific file.

        Args:
            file_ID (int): file identification.
            frequency (int): the audio's sampling frequency.
            path (str): where to find the file.

        Returns:
            fs (int): sampling_rate.
            audio_data (ndarray): audio data.
        
        """
        
        # 'data/RVT/test_files/1.wav' -> ex. de str na lista iterada
        for file in lps_utils.find_files(path):
            # 'data/RVT/test_files', '1.wav'
            complete_path, filename = os.path.split(file)
            
            if '-' in filename:
                # file_ID-frequency.wav
                test_ID = filename.split('-')[0]
                file_freq = (filename.split('-')[1]).split('.')[0]
            else:
                test_ID = filename.split('.')[0]
                file_freq = 8000
                     
            if int(test_ID) == file_ID and int(file_freq) == frequency:
                audio_path = os.path.join(complete_path, filename)
                fs, audio_data = wavfile.read(audio_path)
                break
           
        return fs, audio_data   

