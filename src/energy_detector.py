""" Module providing energy threshold detector. """
from collections import deque
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

        if self.__window_size > len(input_data):
            raise ValueError(f"Window Size ({self.__window_size}) > data ({len(input_data)}).")

        energy = [data**2 for data in input_data]

        detected_samples = []
        energy_sum = 0.0

        # First window
        sample = deque()
        for index in range(self.__window_size):
            sample.append(input_data[index])
            energy_sum += energy[index]

        if energy_sum/self.__window_size >= self.__threshold:
            # [0, self.__window_size-1] -> interval
            detected_samples.append(sample)

        # For next windows
        for index in range(len(input_data) - self.__window_size):
            l = index
            r = index + self.__window_size

            energy_sum -= energy[l]
            energy_sum += energy[r]

            sample.popleft()
            sample.append(input_data[r])

            if energy_sum/self.__window_size >= self.__threshold:
                # [l+1, r] -> interval
                detected_samples.append(sample)

        return np.array(detected_samples)

if __name__ == "__main__":

    detector = EnergyThresholdDetector(10,10)
    manager = ArtifactManager("data/docs/development.csv")
    loader = DataLoader()

    delta = timedelta(seconds=20)

    detected_samples_ = []
    for id_artifact in manager:
        for buoy_id_, time in manager[id_artifact]:

            r_, data_ = loader.get_data(buoy_id_,time-delta,time+delta)
            detection = detector.detect(data_)

            if len(detection) != 0 :
                detected_samples_.append(detection)

    print(len(detected_samples_))
