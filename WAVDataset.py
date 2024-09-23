import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import librosa.display
import sys
import os
from torch.utils.data import Dataset


root = "RVT"
folder = "Dados"

max_freq = 570 

path = os.path.join(root,folder)

class WAVDataset(Dataset):
    
    def __init__( self , path : str , transform=None ) :
        self.data = []
        self.label = []
        self.transform = transform
        
        self.dfs(path)
    
    def __len__( self ) :
        return len(self.data)
    
    def __getitem__( self , index : int ) :
        
        data = self.data[index]
        label = self.data[index]
        
        if self.transform : data = self.transform(data)
        
        return data , label
    
    def dfs( self , folder : str ) :
        subfolders = sorted(os.listdir(folder))
        
        for file in subfolders :
            path = os.path.join(folder,file)
                        
            if  path.endswith('.wav') :         
                self.data.append(read_wav(path))
                self.label.append(read_label(path))

            elif os.path.isdir(path) : 
                self.dfs(path)

            else:
                sys.exit(f'Arquivo indesejado {path}')


# Por favor alguem verifica se essa funcao aq ta certo
# Nao sei se entendi 100% doq vamos considerar como dado e rotulo
def read_wav(path : str , show=False) :
    
    audio, sr = librosa.load(path)

    stft = librosa.stft(audio)

    # Converter para amplitude em dB (opcional)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Transpor a matriz para (tempo, frequência)
    stft_db_transposto = stft_db.T

    if show :
        librosa.display.specshow(stft_db_transposto, sr=sr, x_axis='linear', y_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Espectrograma (Tempo vs Frequência)')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Tempo (s)')
        plt.show()
        
    return stft_db_transposto

def read_label(path : str) : pass

if __name__ == "__main__" : 
    
    read_wav('RVT/Teste/teste.wav',True)