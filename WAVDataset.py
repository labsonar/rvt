import pandas as pd
import numpy as np
import datetime
import os
from scipy.io import wavfile

# Precisamos de uma funcao assim:
# Acessar audio da boiax do dia Y do instante A ateh o instante B

class RVTDataloader():
    
    def __init__(self , root : str) :
        self.root = os.path.join(os.path.abspath(os.getcwd()),root)
        self.data = {}
    
        days = sorted(os.listdir(self.root))
        for day in days:
            path_ = os.path.join(self.root,day)
            
            if not os.path.isdir(path_) : 
                raise FileError(f'Arquivo Indesejado: {path_}')
            
            self.data[day] = {}
            
            boias = sorted(os.listdir(path_))
            for boia in boias :
                path_ = os.path.join(self.root,day,boia)
                
                if not os.path.isdir(path_) : 
                    raise FileError(f'Arquivo Indesejado: {path_}')
                
                self.data[day][boia] = []     
                
                archives = sorted(os.listdir(path_))
                for archive in archives :
                    path_ = os.path.join(self.root,day,boia,archive)
                    
                    if not path_.endswith('.wav') :
                        raise FileError(f'Arquivo Indesejado: {path_}')
                    
                    self.data[day][boia].append(path_)
        
        self.data = pd.DataFrame(self.data)
    
    # Retorna a quantidade de arquivos.wav
    def __len__(self) :
        return self.data.applymap(len).sum().sum()
        
    def acess(self , dia : str , boia : str , a : any , b : any) :
        
        # Antes de tudo, criar um formato para recebermos o dia e a boia
        # Acho melhor converter na hora de acessar os arquivos
        
        # dia = day_format(dia)
        # boia = boia_format(boia)
        
        # concatenar todos os audios entre a e b (qual o formato de a e b?)
        
        data_resp = []
        taxa = None
        file1 = None
        for audio in self.data[dia][boia]:
            
            audio_ = os.path.basename(audio)
            
            # audio_ = transform(audio) # mensurar em numeros para comparar com a e b 
            
            # Posso fazer uma busca binaria pra achar o inicio e o fim (se precisar otimizar depois , ai nao precisa iterar por todos os audios)
            if a <= audio_ <= b : 
                
                taxa_de_amostragem , dados = wavfile.read(audio)
                
                if taxa == None :
                    taxa = taxa_de_amostragem 
                    file1 = audio
                
                elif taxa == taxa_de_amostragem :
                    data_resp.append(dados)
                    
                else : 
                    raise FileError(f'Audios com diferentes taxas de amostragem {file1} e {audio} nao podem ser concatenados')
                
        data_resp = np.concatenate(data_resp)
                
        file_name = f'{generate_file_name()}.wav' # Adicionar caracteristica do acesso ao nome do arquivo (por exemplo, o intervalo [a,b])
        
        out_path = os.path.join(self.root,dia,boia,file_name)
        
        if taxa == None : 
            raise FileError(f'o Arquivo {out_path} nao pode ser gerado')
        
        wavfile.write(out_path , taxa , data_resp)
        
        self.data[dia][boia].append(out_path)
        
        return data_resp

    def export(self) :
        self.data.to_excel()

def generate_file_name():
    
    now = datetime.datetime.now()

    counter = 0
    while os.path.exists(f'{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{counter:04d}.wav') :
        counter+=1

    return f'{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{counter:04d}.wav'


class FileError(Exception) :
    def __init__(self , message : str) :
        super().__init__(message)

if __name__ == "__main__" : 
    
    root = "RVT"
    folder = "Dados"

    path = os.path.join(root,folder)
    
    RVTDataloader(path)
