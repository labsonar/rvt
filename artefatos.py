""" Module providing artifact manager of RVT audio system. """
import typing
import datetime
import pandas as pd

class ArtifactLoader():
    """ Class representing RVT Artifact system. """

    def __init__(self, base_path : str):

        self.path = base_path
        self.data = pd.read_csv(base_path)

    def __len__(self):
        pass

    def artifact_amount(self, artifact_id: int) -> int:
        # faz umas funções de retornem o numero de artefatos
        pass

    def acess_data(self, artifact_id: int) -> typing.Dict[str, object]:
        # e dado um id de artefato acesso os dados
        pass

    def get_map(self, artifact_id: int) -> typing.Dict[str, datetime.datetime]:
        # mas pode retornar um mapa de id da boia para datetime tbm
        pass

    def get_time(self, artifact_id: int, buoy_id: int) -> datetime.datetime:
        """ Get the detection time of the artifact in the determined buoy.

        Args:
            artifact_id (int): Identification of the artifact
            buoy_id (int): Identification of the buoy

        Returns:
            datetime.datetime: Time when the artifact was detected by the buoy
        """
        # TODO Posso acessar diretamente o arquivo em que o artefato foi detectado

        str_date = str(self.data["Dia do Exercício"][artifact_id])
        str_time = self.data[f"Boia{buoy_id}-offset"][artifact_id]

        time = datetime.datetime.strptime( str_date+str_time , "%Y%m%d%H:%M:%S.%f" )

        return time

if __name__ == "__main__":
    store = ArtifactLoader('Artefatos.csv')

    # Pylint reclama do todo, oq eu faco?
