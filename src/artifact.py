""" Module providing artifact manager of RVT audio system. """
import typing
import datetime
import bisect
import pandas as pd

class ArtifactLoader():
    """ Class representing RVT Artifact system. """

    def __init__(self, base_path="Artifacts.csv"):
        self.path = base_path
        self.data = pd.read_csv(base_path)

    def __len__(self) -> int:
        """ Get the amount of diferent artifacts.
        
        Returns:
            int: amount of diferent artifacts.
        """

        return self.data["Tipo de Tiro"].nunique()

    def find_index(self, artifact_id: int) -> int:
        """ Find index from an artifact in a dataframe.

        Args:
            artifact_id (int): Artifact identification.

        Raises:
            KeyError: Artifact id does not exist.

        Returns:
            int: index in dataframe.
        """

        index = bisect.bisect_left(self.data["ID artefato"] , artifact_id)
        if self.data["ID artefato"][index] != artifact_id :
            raise KeyError("Artifact id does not exist")

        return index

    def artifact_amount(self, artifact_type: str) -> int:
        """ Gets the number of same type artifacts.

        Args:
            artifact_type (str): Typo of artifact.

        Raises:
            KeyError: Artifact type do not exist.

        Returns:
            int: Amount of same type artifacts.
        """

        try:
            amount = self.data["Tipo de Tiro"].value_counts()[artifact_type]
        except KeyError as err:
            raise KeyError("Artifact type do not exist.") from err

        return amount

    def acess_data(self, artifact_id: int) -> typing.Dict[str, object]:
        """ Guiven an artifact id, acess all data.

        Args:
            artifact_id (int): artifact identification.

        Returns:
            typing.Dict[str, object]: all data from determined artifact.
        """

        index = self.find_index(artifact_id)

        data = {
            "Index" : index ,
            "ID artefato" : self.data.iloc[index]["ID artefato"] ,
            "Tipo de Tiro" : self.data.iloc[index]["Tipo de Tiro"] ,
            "Falha no Tiro" : bool(self.data.iloc[index]["Falha no Tiro"]) ,
            "Caracterização" : self.data.iloc[index]["Caracterização"] ,
            "Arquivo de Teste" : int(self.data.iloc[index]["Arquivo de Teste"]) ,
            "Offsets" : self.get_time_all(artifact_id) ,
            "Arquivos" : [self.data.iloc[index][f"Boia{i}-Arquivo"] for i in range(1, 6)]
            }

        return data

    def get_time_all(self, artifact_id: int) -> typing.List[datetime.datetime]:
        """ Get the detection time of the artifact in all buoys.

            Args:
                artifact_id (int): Identification of the artifact.

            Returns:
                typing.List[datetime.datetime]: List with the detection time of each buoy.
        """

        artifact_id = self.find_index(artifact_id)

        # TODO quis indexar em 1 pra ficar de forma que mapa[boia_id] = offset de boia id, devo mudar?

        mapa = [None]
        for i in range(1,6):

            try :
                mapa.append(self.get_time(artifact_id,i))
            except KeyError:
                mapa.append(None)

        return mapa

    def get_time(self, artifact_id: int, buoy_id: int) -> datetime.datetime:
        """ Get the detection time of the artifact in the determined buoy.

        Args:
            artifact_id (int): Identification of the artifact.
            buoy_id (int): Identification of the buoy.

        Raises:
            KeyError: artifact_id or buoy_id does not exist.

        Returns:
            datetime.datetime: Time when the artifact was detected by the buoy.
            None: Audio of artifact in buoy do not exist.
        """
        # TODO Posso acessar diretamente o arquivo em que o artefato foi detectado # pylint: disable=fixme

        artifact_id = self.find_index(artifact_id)

        try :
            str_date = str(self.data["Dia do Exercício"][artifact_id])
            str_time = self.data[f"Boia{buoy_id}-offset"][artifact_id]
        except KeyError as out_of_range:
            raise KeyError("artifact_id or buoy_id does not exist") from out_of_range

        try :
            time = datetime.datetime.strptime( str_date+str_time , "%Y%m%d%H:%M:%S.%f" )
        except ValueError :
            return None

        return time

if __name__ == "__main__":
    store = ArtifactLoader()
