""" Module providing artifact sets of development, test and validation. """
import os
from random import shuffle, seed
import typing
from enum import Enum
import pandas as pd
from artifact import ArtifactManager

class Sets(Enum):
    """ Class representing types of sets.
    
        Members:
            DEVELOPMENT: Development set.
            VALIDATE: Validate set.
            TEST: Test set.
    """
    DEVELOPMENT = 0
    VALIDATE = 1
    TEST = 2

    def __str__(self):
        return self.name.lower()

    @classmethod
    def __len__(cls) -> int:
        return len(cls.list())

    @classmethod
    def list(cls) -> typing.List[str]:
        """ Make a list whith all Sets

        Returns:
            list: list with all Sets.
        """
        return list(cls)

class ArtifactSet:
    """ Class representing RVT data set management. """

    def __init__(self, base_path="data/docs/verify.csv", random_state=None):
        self.path = base_path
        self.data = pd.read_csv(base_path).astype(bool)
        self.manager = ArtifactManager()
        self.all_data = []
        self.sets = {}
        seed(random_state)

        for set_ in Sets:
            self.sets[str(set_)] = []

    def restringe_data(self, valid: typing.Callable[[int, int], bool]) -> None:
        """ Selects an subset of base data.

        Args:
            valid (Function(int, int)): Function that returns if audio is valid.
        """

        for id_artifact in self.manager:
            for buoy_id, time in self.manager[id_artifact]:
                if valid(id_artifact, buoy_id):
                    self.all_data.append((id_artifact, buoy_id, time))

    def generate_sets(self, development: int, validate: int, test: int) ->\
        None:
        """ Destribute data in 3 sets.

        Args:
            development (int): parts to development.
            test (int): part to test.
            validate (int): part to validate.            
        """
        total = development + validate + test

        development = development/total
        validate = validate/total
        test = test/total

        shuffle(self.all_data)
        for index , item in enumerate(self.all_data):

            if index <= development*len(self.all_data):
                self.sets[str(Sets.list()[0])].append(item)

            elif index <= (validate+development)*len(self.all_data):
                self.sets[str(Sets.list()[1])].append(item)

            else:
                self.sets[str(Sets.list()[2])].append(item)

    def all_3(self, artifact_id: int, buoy_id: int) -> bool:
        """ Verify if audio is centralized, audible and visible 

        Args:
            artifact_id (int): Id of artifact.
            buoy_id (int): Id of buoy.

        Returns:
            bool: if audio from artifact and buoy is valid.
        """

        index = self.manager.find_index(artifact_id)

        visible = self.data.iloc[index][f'Buoy {buoy_id} - Visible']
        centralized = self.data.iloc[index][f'Buoy {buoy_id} - Centralized']
        audible = self.data.iloc[index][f'Buoy {buoy_id} - Audible']

        return visible and centralized and audible

    def review(self) -> None:
        """ View data destribuition in all subsets.
        """

    def save(self) -> None:
        """ Save subsets in .csv files.
        """

        path = "data/docs"

        for subset , audios in self.sets.items():
            file_name = f"{subset}.csv"
            file_dict = {}

            for artifact_id, buoy_id, time in audios:

                column = f"Buoy{buoy_id}-Time"

                if not column in file_dict:
                    file_dict[column] = {}

                file_dict[column][artifact_id] = time

            file = pd.DataFrame.from_dict(file_dict, orient='columns')
            file.index.name = "Artifact ID"
            file.sort_values(by="Artifact ID", ascending=True, inplace=True)
            file.to_csv(os.path.join(path,file_name))

if __name__ == "__main__":
    Set  = ArtifactSet(random_state=1042)

    Set.restringe_data(Set.all_3)
    Set.generate_sets(0.8,0.1,0.1)

    for name, data in Set.sets.items():
        print(f"Subset {name} - {100*len(data)/len(Set.all_data) :.2f}% of total data")

    Set.save()

    dev = pd.read_csv("data/docs/development.csv")
    tes = pd.read_csv("data/docs/test.csv")
    val = pd.read_csv("data/docs/validate.csv")

    print(dev)
    print(tes)
    print(val)
