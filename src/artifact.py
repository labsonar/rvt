""" Module providing artifact manager of RVT audio system. """
import typing
import datetime
import bisect
import pandas as pd

class ArtifactManager():
    """ Class representing RVT Artifact system. """

    def __init__(self, base_path="data/artifacts.csv"):
        self.path = base_path
        self.data = pd.read_csv(base_path)
        self.index = 0

    def __len__(self) -> int:
        """ Get the amount of artifacts.
        
        Returns:
            int: amount of artifacts.
        """
        return len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            value = Artifact(self.data["Artifact ID"][self.index])
            self.index += 1
            return value
        raise StopIteration

    def id_from_type(self, artifact_type: str) -> typing.List[int]:
        """ Gets all artifacts id's of determined type

        Args:
            artifact_type (str): Type of the artifact.

        Returns:
            typing.List[int]: List with the id's of the artifacts whith same type
        """

        return self.data[self.data['Shooting Type'] == artifact_type]['Artifact ID'].tolist()

    def find_index(self, artifact_id: int) -> int:
        """ Find index from an artifact in a dataframe.

        Args:
            artifact_id (int): Artifact identification.

        Raises:
            KeyError: Artifact ID does not exist.

        Returns:
            int: index in dataframe.
        """

        index = bisect.bisect_left(self.data["Artifact ID"] , artifact_id)
        if self.data["Artifact ID"][index] != artifact_id :
            raise KeyError("Artifact ID does not exist")

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

        # TODO Professor pediu pra mudar aq, mas eu n entendi # pylint: disable=fixme

        try:
            amount = self.data["Tipo de Tiro"].value_counts()[artifact_type]
        except KeyError as err:
            raise KeyError("Artifact type do not exist.") from err

        return amount

    def get_time_all(self, artifact_id: int) -> typing.Dict[str, datetime.datetime]:
        """ Get the detection time of the artifact in all buoys.

            Args:
                artifact_id (int): Identification of the artifact.

            Returns:
                typing.Dict[str, datetime.datetime]: Map with the detection time of each buoy.
        """

        mapa = {}
        for i in range(1,6):

            try :
                mapa[f"Buoy{i}"] = self.get_time(artifact_id,i)
            except ValueError :
                continue

        return mapa

    def get_time(self, artifact_id: int, buoy_id: int) -> datetime.datetime:
        """ Get the detection time of the artifact in the determined buoy.

        Args:
            artifact_id (int): Identification of the artifact.
            buoy_id (int): Identification of the buoy.

        Raises:
            KeyError: artifact_id or buoy_id does not exist.
            ValueError: Buoy did not record the artifact.

        Returns:
            datetime.datetime: Time when the artifact was detected by the buoy.
        """
        index = self.find_index(artifact_id)

        try :
            str_day = str(self.data["Exercise Day"][index])
            str_file = str(self.data[f"Buoy{buoy_id}-File"][index])[2:-10]
            str_offset = str(self.data[f"Buoy{buoy_id}-Offset"][index])

        except KeyError as out_of_range:
            raise KeyError("artifact_id or buoy_id does not exist") from out_of_range

        try :
            # TODO Revisar isso aq, n sei se entendi a estrutura das datas na planilha # pylint: disable=fixme
            date = datetime.datetime.strptime(str_day+str_file, "%Y%m%d_%H_%M")
            offset_ = datetime.datetime.strptime(str_offset , "%H:%M:%S.%f")
        except ValueError as err :
            raise ValueError(f"Buoy{buoy_id} did not record the artifact {artifact_id}.") from err

        date = date + datetime.timedelta(
                hours=offset_.hour,
                minutes=offset_.minute,
                seconds=offset_.second,
                microseconds=offset_.microsecond
            )

        return date

class Artifact(ArtifactManager):
    """ Class representing an artifact. """

    def __init__(self, artifact_id: int):
        super().__init__()

        # TODO esses atributos sao o suficiente ou eh bom ter mais informacao? # pylint: disable=fixme

        self.iteration = 0
        self.index = self.find_index(artifact_id)
        self.id = artifact_id
        self.type = self.data.iloc[self.index]["Shooting Type"]
        self.failure = bool(self.data.iloc[self.index]["Shooting Failure"])
        self.times = self.get_time_all(artifact_id)
        self.keys = list(self.times.keys())

    def __len__(self):
        return len(self.times)

    def __iter__(self):
        self.iteration = 0
        return self

    def __next__(self):
        if self.iteration < len(self.times):
            key = self.keys[self.iteration]
            value = self.times[key]
            self.iteration += 1
            return key , value
        raise StopIteration

if __name__ == "__main__":

    artifact_manager = ArtifactManager()

    for artifact in artifact_manager:
        for buoy_id_, time in artifact:
            print(f"[{buoy_id_}: {time}]")
