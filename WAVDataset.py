import pandas as pd
import numpy as np
import bisect # busca binaria pra implementada
import sys
import os
from scipy.io import wavfile

# Precisamos de uma funcao assim:
# Acessar audio da boiax do dia Y do instante A ateh o instante B

class RVTDataloader():
    
    def __init__(self , root : str) :
        self.root = root
        self.data = {}
    
        days = sorted(os.listdir(root))
        for day in days:
            path_ = os.path.join(root,day)
            
            if not os.path.isdir(path_) : 
                raise JunkFile(path_)
            
            self.data[day] = {}
            
            boias = sorted(os.listdir(path_))
            for boia in boias :
                path_ = os.path.join(root,day,boia)
                
                if not os.path.isdir(path_) : 
                    raise JunkFile(path_)
                
                self.data[day][boia] = []     
                
                archives = sorted(os.listdir(path_))
                for archive in archives :
                    path_ = os.path.join(root,day,boia,archive)
                    
                    if not path_.endswith('.wav') :
                        raise JunkFile(path_)
                    
                    self.data[day][boia].append(path_)
        
        self.data = pd.DataFrame(self.data)
        
    # Retorna a quantidade de arquivos.wav
    def __len__(self) :
        return self.data.applymap(len).sum().sum()
        
    def acess(self , dia : str , boia : str , a : any , b : any) :
        
        # Antes de tudo, criar um formato para recebermos o dia e a boia
        # podemos converter isso no construtor, ou na hora de acessar os arquivos
        
        # concatenar todos os audios entre a e b
        for audio in self.data[dia][boia]:
            
            audio_ = os.path.basename(audio)
            
            #audio_ = transform(audio) # mensurar em numeros 
            if a <= audio_ <= b : 
                taxa_de_amostragem , dados = wavfile.read(audio)
                #concatenate(dados)

class JunkFile(Exception):
    def __init__(self , path : str) :
        super().__init__(f'Arquivo Indesejado: {path}')

if __name__ == "__main__" : 
    
    root = "RVT"
    folder = "Dados"

    path = os.path.join(root,folder)
    
    RVTDataloader(path)
