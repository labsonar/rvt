''' Dataloader para gerenciar o sistema de audios do RVT '''
import os
import datetime
import typing
import bisect
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

    def get_data(self, buoy_id: int, start_time: datetime.datetime, end_time: datetime.datetime):
        '''
            Funcao para acessar e concatenar dados em um intervalo de tempo para determinada boia
        '''

        data_resp = []
        taxa = None

        # Duas buscas binarias
        # Se o elemento nao for encontrado na busca binaria, eh encontrado um valor maior
        start = bisect.bisect_left(self.file_dict[buoy_id], (start_time, ""))
        end = bisect.bisect_left(self.file_dict[buoy_id], (end_time, ""))
        end-=1 # assim garanto que o indice end esta na lista original

        for index in range(start,end):

            audio_path = self.file_dict[buoy_id][index][1]

            taxa_de_amostragem , dados = wavfile.read(audio_path)

            if not taxa :
                taxa = taxa_de_amostragem
            elif taxa != taxa_de_amostragem :
                dados = decimate(dados, taxa_de_amostragem/taxa) # revisar isso aqui

            data_resp.append(dados)

        # Depois do intervalo geral, vamos fixar as pontas

        if self.file_dict[buoy_id][start][0] > start_time:

            if start==0 :
                raise ValueError("Nao existem dados no intervalo de tempo inteiro")

            time , audio_path = self.file_dict[buoy_id][start-1]
            taxa_de_amostragem , dados = wavfile.read(audio_path)

            if not taxa :
                taxa = taxa_de_amostragem
            elif taxa != taxa_de_amostragem :
                dados = decimate(dados, taxa_de_amostragem/taxa) # revisar isso aqui

            index = abs(start_time.second - time.second) * taxa
            if index < 0 or index >= len(dados):
                raise ValueError("Erro de conta no get_data()")

            data_resp.insert(0,dados[index:])

        if self.file_dict[buoy_id][end][0] < end_time:

            if end==len(self.file_dict[buoy_id]) :
                raise ValueError("Nao existem dados no intervalo de tempo inteiro")

            time , audio_path = self.file_dict[buoy_id][end+1]
            taxa_de_amostragem , dados = wavfile.read(audio_path)

            if not taxa :
                taxa = taxa_de_amostragem
            elif taxa != taxa_de_amostragem :
                dados = decimate(dados, taxa_de_amostragem/taxa) # revisar isso aqui

            index = abs(end_time.second - time.second) * taxa
            if index < 0 or index >= len(dados):
                raise ValueError("Erro de conta no get_data()")

            data_resp.append(dados[:index])

        return taxa , np.concatenate(data_resp)

if __name__ == "__main__":
    data = DataLoader("./Data/RVT/raw_data")

    start_ = datetime.datetime(2024, 1, 19, 13, 41, 0)
    end_ = datetime.datetime(2024, 1, 19, 13, 45, 0)

    taxa_ , data_ = data.get_data(2,start_,end_)

    wavfile.write('teste.wav',taxa_,data_)
    