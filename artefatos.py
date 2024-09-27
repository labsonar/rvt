''' Script para identificar os artefatos '''
import typing
import datetime
import pandas as pd

class Artefatos():
    ''' Classe para gerenciar o Artefatos.ods '''

    def __init__(self, base_path : str, create=False):

        if create:
            self.df = pd.DataFrame.from_dict(Artefatos.get_xlsx_dict(base_path))
        elif base_path.endswith('.xlsx') :
            self.df = pd.read_excel(base_path)
        else :
            raise ValueError("Path can not be read")

        #self.df = self.df.dropna()
        self.path = base_path


        for artefato, buoy_dict in self.df.items():
            print(artefato)
            for buoy_id , lista in buoy_dict.items():
                print('\t Boia', buoy_id)
                for artefato , detecao in lista:
                    print('\t\t', f'{artefato} -> {detecao}')


    @staticmethod
    def get_xlsx_dict(base_path : str) -> \
        typing.Dict[str, typing.Dict[int, \
            typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]]]:
        ''' 
            Organizo o .xlsx para transformar em .csv 

            Abri o .ods no google planilhas e de la exportei como .xslx
            MUITO mal otimizado, mas nao precisa ser otimizado aqui
        '''

        df = pd.read_excel(base_path , sheet_name="compiled_data")
        resp = {}

        for index in range(1,len(df["Unnamed: 1"].dropna())):
            tipo_de_tiro = df["Unnamed: 1"][index]

            if not tipo_de_tiro in resp:
                resp[tipo_de_tiro] = {}

            for jndex in range(1,6) :
                boia = f'Boia {jndex}'
                dia = str(df['Unnamed: 0'][index])

                hora_artefato = df[boia][index]
                hora_detecao = df[f'Unnamed: {5 + 2*jndex}'][index]

                if not jndex in resp[tipo_de_tiro] :
                    resp[tipo_de_tiro][jndex] = []

                offset_artefato = sla(dia,hora_artefato)
                offset_detecao = sla(dia,hora_detecao)

                resp[tipo_de_tiro][jndex].append((offset_artefato,offset_detecao))

        return resp

    def save(self) -> None:
        '''
            exporta os dados para csv 
        '''
        self.df.to_excel(datetime.datetime.now().strftime("%m_%d-")+self.path)

def sla(dia :str, hora : str) -> datetime.datetime:
    if not hora in ["-"," - ","x"] :
        pass
        # transformar hora_detecao em datetime.time 
        #offset_detecao = datetime.datetime(int(dia[:4]), int(dia[4:6]), int(dia[6:]), \
        #    hora_detecao.hour, hora_detecao.minute, \
        #        hora_detecao.second, hora_detecao.microsecond)
    return None

if __name__ == "__main__":
    store = Artefatos('Artefatos.xlsx',True)
