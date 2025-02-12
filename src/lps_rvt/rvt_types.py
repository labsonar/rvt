""" Type Module
"""
import enum

class Ammunition(enum.Enum):
    """ Type of Ammunition """
    EXSUP = "EX-SUP"
    HE3M = "HE3m"
    GAE = "GAE"

    def __str__(self) -> str:
        return str(self.value)

class Subset(enum.Enum):
    """ Type of data subsets """
    TRAIN = 0
    VALIDATION = 1
    VAL = 1
    TEST = 2
