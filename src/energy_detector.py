""" Module providing energy threshold detector. """
from datetime import timedelta
import numpy as np
from artifact import ArtifactManager
from loader import DataLoader
from detector import Detector

class EnergyThresholdDetector(Detector):
    """ Class representing an energy threshold detector. """

    def __init__(self, threshold: float, window_size: int):
        self.__threshold = threshold
        self.__window_size = window_size

    def detect(self, input_data: np.array) -> np.array:
        """
        Args:
            input_data (np.array): vetor de dados

        Returns:
            np.array: amostras onde ocorrem as detecções
        """

        detected_samples = []

        for index in range(len(input_data)-self.__window_size):
            energy_sum = 0
            sample = []

            for jndex in range(self.__window_size):
                data = input_data[index+jndex]
                energy_sum += abs(data)**2
                sample.append(data)

            if energy_sum/self.__window_size >= self.__threshold:
                print(sample)
                detected_samples.append(np.array(sample))

        return np.array(detected_samples)

if __name__ == "__main__":

    detector = EnergyThresholdDetector(0.0000,1000)
    manager = ArtifactManager("data/docs/development.csv")
    loader = DataLoader()

    delta = timedelta(seconds=20)

    for id_artifact in manager:
        for buoy_id_, time in manager[id_artifact]:

            data_ = loader.get_data(buoy_id_,time-delta,time+delta)
            detected_samples_ = detector.detect(data_)

    print(detected_samples_)
