''' Dataloader para gerenciar o sistema de audios do RVT '''
import os
import datetime
import typing

import lps_utils.utils as lps_utils

class DataLoader():
    ''' Dataloader para gerenciar os audios providos pelas boias '''

    def __init__(self, base_path: str) -> None:
        self.file_dict = DataLoader.get_file_dict(base_path)

        for buoy_id, file_list in self.file_dict.items():
            print(buoy_id)
            for file in file_list:
                print('\t: ', file[0], " -> ", file[1])

    @staticmethod
    def get_file_dict(base_path: str) -> \
        typing.Dict[int, typing.List[typing.Tuple[datetime.datetime, str]]]:
        ''' 
            Funcao para auxiliar o construtor da classe.
            Transforma uma pasta de arquivos em um dataframe
        '''
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


    def get_data(self, buoy_id: int, start_time: datetime.datetime, end_time: datetime.datetime,
                decimate=None):
        '''
            Funcao para acessar e concatenar dados em um intervalo de tempo para determinada boia
        '''


if __name__ == "__main__":
    DataLoader("./Data/RVT/raw_data")
    