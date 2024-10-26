""" Module providing artifact manager of RVT audio system. """
import typing
import datetime
import bisect
import pandas as pd

class ArtifactManager():
    """ Class representing RVT Artifact system. """

    def __init__(self, base_path="data/docs/artifacts.csv"):
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
        return iter(self.data["Artifact ID"].to_list())

    def __getitem__(self, artifact_id: int):
        return self.get_time_all(artifact_id).items()

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
            amount = self.data["Shooting Type"].value_counts()[artifact_type]
        except KeyError as err:
            raise KeyError("Artifact type do not exist.") from err

        return amount

    def get_time_all(self, artifact_id: int) -> typing.Dict[int, datetime.datetime]:
        """ Get the detection time of the artifact in all buoys.

            Args:
                artifact_id (int): Identification of the artifact.

            Returns:
                typing.Dict[int, datetime.datetime]: Map with the detection time of each buoy.
        """

        mapa = {}
        for i in range(1,6):

            try :
                mapa[i] = self.get_time(artifact_id,i)
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

if __name__ == "__main__":

    manager = ArtifactManager()

    for id_artifact in manager:
        print(f"Artifact: {id_artifact}")
        for buoy_id_, time in manager[id_artifact]:
            print(f"\t[{buoy_id_}: {time}] ")
