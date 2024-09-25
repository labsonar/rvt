''' Dataloader para gerenciar o sistema de audios do RVT '''
import os
import datetime
import typing
import numpy as np
from scipy.io import wavfile # Nao sei se tem alguma biblioteca no Lps_utils que recebe audios

import lps_utils.utils as lps_utils
from lps_sp.signal import decimate

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
                decimate_num=None):
        '''
            Funcao para acessar e concatenar dados em um intervalo de tempo para determinada boia
        '''

        data_resp = []
        taxa = decimate_num
        for time , audio_path in self.file_dict[buoy_id]:

            # Posso fazer uma busca binaria pra achar o inicio e o fim na lista
            # (se precisar otimizar depois , ai nao precisa iterar por todos os audios)

            if start_time <= time <= end_time :

                taxa_de_amostragem , dados = wavfile.read(audio_path)

                if decimate_num :
                    dados = decimate(dados, decimate_num)
                elif not taxa :
                    taxa = taxa_de_amostragem
                elif taxa != taxa_de_amostragem :
                    dados = decimate(dados, taxa_de_amostragem/taxa) # revisar isso aqui

                data_resp.append(dados)

        data_resp = np.concatenate(data_resp)

        return taxa , data_resp


if __name__ == "__main__":
    data = DataLoader("./Data/RVT/raw_data")

    start = datetime.datetime(2024, 1, 19, 13, 41, 0)
    end = datetime.datetime(2024, 1, 19, 13, 45, 0)

    taxa_ , data_ = data.get_data(2,start,end)

    wavfile.write('teste.wav',taxa_,data_) # Nao consigo ouvir o audio, talvez eu tenha errado
    