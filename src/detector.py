""" Module providing abstract detector. """
import typing
import abc
import numpy as np

class Detector(abc.ABC):
    """ Abstract detector for acoustical events. """

    @abc.abstractmethod
    def detect(self, input_data: np.array) -> np.array:
        """
        Args:
            input_data (np.array): vetor de dados

        Returns:
            np.array: amostras onde ocorrem as detecções
        """

    def evaluate(self,
                input_data: np.array,
                expected_detections: typing.List[int],
                tolerance: int) -> np.array:
        """Avaliação da detecção

        Args:
            input_data (np.array): vetor de dados
            expected_detections (typing.List[int]): \
                                amostras onde ocorrem os eventos a serem detectados 
            tolerance (int): offset de amostras considerada aceitavel

        Returns:
            np.array: [True_positive, False Negative;
                        False positive, True Negative]
        """
