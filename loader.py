""" Module providing Dataloader to manage RVT audio system. """
import os
import datetime
import typing
import bisect
import numpy as np
from scipy.io import wavfile # Nao sei se tem alguma biblioteca no Lps_utils que recebe audios

import lps_utils.utils as lps_utils
from lps_sp.signal import decimate

class DataLoader():
    """ Class representing the RVT audio system. """

    def __init__(self, base_path: str) -> None:
        self.file_dict = DataLoader.get_file_dict(base_path)

        # for buoy_id, file_list in self.file_dict.items():
        #     print(buoy_id)
        #     for file in file_list:
        #         print('\t: ', file[0], " -> ", file[1])

    @staticmethod
    def get_file_dict(base_path: str) -> \
        typing.Dict[int, typing.List[typing.Tuple[datetime.datetime, str]]]:
        """ Function to retrieve and organize data from a folder.

            Args:
                base_path (str): Path to the folder.
        """

        file_dict = {}

        for file in lps_utils.find_files(base_path):

            complete_path, filename = os.path.split(file)
            path_list = complete_path.split(os.sep)

            if not path_list[-1].startswith('boia'):
                continue

            full_day = path_list[-2]
            try:
                datetime.datetime.strptime(full_day, "%Y%m%d")
            except ValueError:
                continue


            buoy_id = int(path_list[-1][4:])

            filename_list = filename.split('_')
            hour = filename_list[1]
            minute = filename_list[2]

            date = datetime.datetime.strptime(f"{full_day}{hour}{minute}", "%Y%m%d%H%M")

            if buoy_id not in file_dict:
                file_dict[buoy_id] = []

            file_dict[buoy_id].append((date,file))


        file_dict = {k: file_dict[k] for k in sorted(file_dict)}

        for buoy_id, file_list in file_dict.items():
            file_dict[buoy_id] = sorted(file_list)

        return file_dict

    def get_data(self, buoy_id: int, \
                start_time: datetime.datetime, end_time: datetime.datetime) -> \
                typing.Tuple[int, np.ndarray[np.int16]]:
        """ Concatenate audios with precision in the desired interval.

            Args: 
                buoy_id (int): Buoy identification. 
                start_time (datetime.datetime): Start time of the desired audio result.
                end_time (datetime.datetime): End time of the desired audio result.

            Raises: 
                ValueError: Requested date and time out of range.

            Returns: 
                int: Sampling rate.
                ndarray: Audio data.
        """

        # print("\tDesired times: ", start_time, " -> ", end_time)

        data_resp = []
        taxa = None

        # Binary search to find where this date and time
        # should be inserted to keep the list in order
        start = bisect.bisect_left(self.file_dict[buoy_id], (start_time, ""))
        end = bisect.bisect_left(self.file_dict[buoy_id], (end_time, ""))

        if start == 0 or end == len(self.file_dict[buoy_id]):
            raise ValueError("Requested date and time out of range")

        start -= 1
        # Starting index pointing to the first file with data above the requested start time.
        # Therefore, the request time is part of the previous file

        # print("\tIndexes: ", start, " -> ", end)
        # print("\tRecorded files: ", self.file_dict[buoy_id][start][1], \
        #       " -> ", self.file_dict[buoy_id][end][1])

        for index in range(start,end):

            audio_path = self.file_dict[buoy_id][index][1]

            taxa_de_amostragem , dados = wavfile.read(audio_path)

            if not taxa :
                taxa = taxa_de_amostragem
            elif taxa != taxa_de_amostragem :
                dados = decimate(dados, taxa_de_amostragem/taxa) # TODO revisar isso aqui

            data_resp.append(dados)

        data_resp = np.concatenate(data_resp)

        delta = start_time - self.file_dict[buoy_id][start][0]
        start_overhead = int(delta.total_seconds() * taxa)
        data_resp = data_resp[start_overhead:]

        desired_n_samples = int((end_time - start_time).total_seconds() * taxa)

        if len(data_resp) < desired_n_samples:
            # True only if the desired data range is not continuously recorded in the raw files
            raise ValueError("Requested date and time out of range")

        data_resp = data_resp[:desired_n_samples]

        print(type(data_resp[0]))

        return taxa, data_resp

if __name__ == "__main__":
    data = DataLoader("./Data/RVT/raw_data")
