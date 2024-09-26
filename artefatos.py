''' Script para identificar os artefatos '''
import pandas as pd

class Artefatos():
    ''' Classe para gerenciar o Artefatos.ods '''

    def __init__(self, base_path : str):
        self.file_data = pd.read_csv(base_path)

    def find(self, artefato : str ):
        ''' Funcao para retornar os intervalos de tempo em que existe um artefato '''

def transformar_em_csv(base_path : str):
    ''' receber o .ods e fazer virar um csv '''

if __name__ == "__main__":

    data = Artefatos("Artefatos.ods")

    # artefatos = [  ]
    data.find("")
    