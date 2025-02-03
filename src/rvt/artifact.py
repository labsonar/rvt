""" Module providing artifact manager of RVT audio system. """
import typing
import datetime
import bisect
import re
import pandas as pd

class ArtifactManager():
    """ Class representing RVT Artifact system. """

    def __init__(self, base_path="data/docs/artifacts.csv"):
        self.path = base_path
        self.data = pd.read_csv(base_path)
        self.index = 0

        for artifact_id in self:
            for buoy_id in range(1,6):

                try:
                    self.__set_time(artifact_id, buoy_id)
                except ValueError:
                    continue
                except KeyError:
                    continue

    def __len__(self) -> int: # TODO change to give the amount of pairs {artifact ID, buoy ID} exist in class #pylint: disable=fixme
        """ Get the amount of valuable data.
        
        Returns:
            int: amount of pairs {artifact, buoy}.
        """

        columns1 = ["Buoy1-Time","Buoy4-Time","Buoy3-Time","Buoy2-Time","Buoy5-Time"]
        columns = [col for col in columns1 if col in self.data.columns]

        if len(columns):
            return self.data[columns].notna().sum().sum()

        columms2 = ["Buoy1-File","Buoy2-File","Buoy3-File","Buoy4-File","Buoy5-File"]
        columms3 = ["Buoy1-Offset","Buoy2-Offset","Buoy3-Offset","Buoy4-Offset","Buoy5-Offset"]
        amount = self.data[columms2].notna().sum().sum()
        assert(amount == self.data[columms3].notna().sum().sum())
        return amount

    def __iter__(self):
        return iter(self.data["Artifact ID"].to_list())

    def __getitem__(self, artifact_id: int) -> int:
        return self.get_time_all(artifact_id).keys()

    def __set_time(self, artifact_id: int, buoy_id: int) -> None:
        """ Set time of pair artifact, buoy in database.

        Args:
            artifact_id (int): Identification of the artifact.
            buoy_id (int): Identification of the buoy.

        Raises:
            KeyError: artifact_id or buoy_id does not exist.
            ValueError: Buoy did not record the artifact.
        """

        index = self.find_index(artifact_id)

        if f"Buoy{buoy_id}-Time" in self.data.columns:
            date = self.data[f"Buoy{buoy_id}-Time"][index]
            if isinstance(date, str):
                self.data.loc[index, f"Buoy{buoy_id}-Time"] = \
                    datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
                return

        if not f"Buoy{buoy_id}-File" in self.data.columns:
            raise KeyError("artifact_id or buoy_id does not exist")

        str_day = str(self.data["Exercise Day"][index])
        str_file = str(self.data[f"Buoy{buoy_id}-File"][index])[2:-10]
        str_offset = str(self.data[f"Buoy{buoy_id}-Offset"][index])

        try :
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

        if not f"Buoy{buoy_id}-Time" in self.data.columns:
            self.data[f"Buoy{buoy_id}-Time"] = pd.NaT

        self.data.loc[index, f"Buoy{buoy_id}-Time"] = date

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

    def artifact_amount_by_type(self, artifact_type: str) -> int:
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

    def artifact_amount_by_time(self, buoy_id: int, \
        start: datetime.datetime, end: datetime.datetime) -> typing.List[int]:
        """ Gets the amount of artifacts in time interval.

        Args:
            start (datetime.datetime): Start time of interval.
            end (datetime.datetime): End time of interval.

        Returns:
            int: Amount of artifacts in time interval.
        """

        column = f"Buoy{buoy_id}-Time"

        if not column in self.data.columns:
            raise ValueError(f"Buoy {buoy_id} do not exist.")

        filtered = self.data[
            (self.data[column] >= start) &
            (self.data[column] <= end)
        ]

        return filtered[column].tolist()

    def get_time_all(self, artifact_id: int) -> typing.Dict[int, datetime.datetime]:
        """ Get the detection time of the artifact in all buoys.

            Args:
                artifact_id (int): Identification of the artifact.

            Returns:
                typing.Dict[int, datetime.datetime]: Map with the detection time of each buoy.
        """

        mapa = {}
        for i in range(1,6):

            try:
                mapa[i] = self.get_time(artifact_id, i)
            except ValueError:
                continue
            except KeyError:
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

        if not f"Buoy{buoy_id}-Time" in self.data.columns:
            raise ValueError("Buoy Id do not exist.")

        date = self.data[f"Buoy{buoy_id}-Time"][index]

        if pd.isna(date):
            raise ValueError("Record of artifact in buoy Id do not exist.")

        if isinstance(date, str):
            return datetime.datetime.strptime(self.data[f"Buoy{buoy_id}-Time"][index],\
                "%Y-%m-%d %H:%M:%S.%f")

        if isinstance(date, datetime.datetime):
            return date

        raise ValueError("Unexpected error.")

    def get_types(self) -> typing.List[str]:
        """ get all types of artifact in data.

        Returns:
            typing.List[str]: All artifact types.
        """

        return list(self.data["Shooting Type"].unique())

    def get_buoys(self) -> typing.List[int]:
        """ Get all buoys stored in data.

        Returns:
            typing.List[int]: All available Buoys.
        """

        pattern = r"Buoy\d-File"
        asw = []
        for column in self.data.columns:
            if re.fullmatch(pattern, column):
                asw.append(int(column[4]))
        return asw

    def type_from_id(self, artifact_id: int) -> str:
        """ Get type of the artifact by its ID

        Args:
            artifact_id (int): Artifact Identification

        Returns:
            str: Artifact Type
        """

        index = self.find_index(artifact_id)
        return self.data.loc[index, "Shooting Type"]

if __name__ == "__main__":

    manager = ArtifactManager("data/docs/test_artifacts.csv")

    for id_artifact in manager:
        print(f"Artifact: {id_artifact}")
        for buoy_id_ in manager[id_artifact]:
            print(f"\t[{buoy_id_}] {manager.get_time(id_artifact, buoy_id_)} ")

    # start_ = datetime.datetime(2023, 9, 12, 16, 20)
    # end_ = datetime.datetime(2023, 9, 12, 17, 40)

    # print(manager.artifact_amount_by_time(3, start_, end_))
