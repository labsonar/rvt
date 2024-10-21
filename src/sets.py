""" Module providing artifact sets of development, test and validation. """
from random import shuffle
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
        return self.name.capitalize()

    @classmethod
    def __len__(cls) -> int:
        return len(cls.list())

    @classmethod
    def list(cls) -> typing.List:
        """ Make a list whith all Sets

        Returns:
            list: list with all Sets.
        """
        return list(cls)

class ArtifactSet:
    """ Class representing RVT data set management. """

    def __init__(self, base_path="data/verify.csv"):
        self.path = base_path
        self.data = pd.read_csv(base_path)

    @staticmethod
    def generate_sets(development: int, validate: int, test: int, base_data: typing.List) -> \
                        typing.Dict[str, typing.List]:
        """ Destribute data in 3 sets.

        Args:
            development (int): parts to development.
            test (int): part to test.
            validate (int): part to validate.
            
        Returns:
            Dict[str, List] = Dict whith list of elements of all sets.
            
        """

        total = development + validate + test

        development = development/total
        validate = validate/total
        test = test/total

        shuffle(base_data)

        sets_dict = {}
        for set_ in Sets:
            sets_dict[set_] = []

        for index , item in enumerate(base_data):

            if index <= development*len(base_data):
                sets_dict[Sets.list()[0]].append(item)

            elif index <= validate*len(base_data):
                sets_dict[Sets.list()[1]].append(item)

            else:
                sets_dict[Sets.list()[2]].append(item)

        return sets_dict

    @staticmethod
    def restringe_data(valid: typing.Callable[[int, int], bool]) -> typing.List[int, int]:
        """ Selects an subset of base data.

        Args:
            valid (Function(int, int)): Function that returns if audio is valid.

        Returns:
            typing.List: Restringed subset of base data.
        """

        manager = ArtifactManager()
        base_data = []

        for id_artifact in manager:
            for buoy_id, time in manager[id_artifact]: #pylint: disable=fixme
                if valid(id_artifact, buoy_id):
                    base_data.append((id_artifact, buoy_id))

        return base_data

    def all_3(self, artifact_id: int, buoy_id: int) -> bool:
        """ Verify if audio is centralized, audible and visible 

        Args:
            artifact_id (int): Id of artifact.
            buoy_id (int): Id of buoy.

        Returns:
            bool: if audio from artifact and buoy is valid.
        """
