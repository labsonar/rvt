""" Module providing energy threshold detector. """
import typing
import numpy as np

from src.detector import Detector

class TestDetector(Detector):
    """ Class representing a test detector. """

    def __init__(self, boolean: bool= True):
        self.__boolean = boolean
        self.name = f"Always {boolean} Detector"

    def detect(self, input_data: np.array) -> typing.Tuple[typing.List[int], int]:
        """
        Args:
            input_data (np.array): Data vector.

        Returns:
            typing.Tuple[typing.List[int], int]: Detected samples and number of windows.
        """

        detected_samples = []

        for i in range(len(input_data)):
            if self.__boolean:
                detected_samples.append(i)
        
        return detected_samples, len(input_data)