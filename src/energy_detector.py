""" Module providing energy threshold detector. """
import typing
from datetime import timedelta
import numpy as np
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from artifact import ArtifactManager
from loader import DataLoader
from detector import Detector

class EnergyThresholdDetector(Detector):
    """ Class representing an energy threshold detector. """

    def __init__(self, threshold: float, window_size: int, \
            scaler: typing.Optional[TransformerMixin] = None):
        self.__threshold = threshold
        self.__window_size = window_size
        self.__scaler = scaler

    def detect(self, input_data: np.array) -> np.array:
        """
        Args:
            input_data (np.array): vetor de dados

        Returns:
            np.array: amostras onde ocorrem as detecções
        """

        if self.__window_size > len(input_data):
            raise ValueError(f"Window Size ({self.__window_size}) > data ({len(input_data)}).")

        if self.__scaler:
            input_data = self.__scaler.fit_transform(input_data.reshape(-1, 1)).flatten()

        detected_samples = []
        energy_sum = 0.0

        # First window
        energy_sum = np.sum(input_data[:self.__window_size]**2)

        # For next windows
        detection_occuring = False
        for index in range(len(input_data) - self.__window_size):
            l = index
            r = index + self.__window_size

            mean = energy_sum/self.__window_size
            if input_data[r]*input_data[r] > self.__threshold * mean:

                if not detection_occuring:
                    detection_occuring = True
                    detected_samples.append(r)

            elif detection_occuring:
                detection_occuring = False

            energy_sum -= input_data[l]*input_data[l]
            energy_sum += input_data[r]*input_data[r]

        return np.array(detected_samples)

def main():
    """ Main for debuging. """
    scalar = preprocessing.MinMaxScaler()

    detector = EnergyThresholdDetector(3.5,1600,scalar)
    manager = ArtifactManager("data/docs/development.csv")
    loader = DataLoader()

    delta = timedelta(seconds=20)

    detected_samples = []
    count = 1
    for id_artifact in manager:
        for buoy_id_, time in manager[id_artifact]:

            r , data_ = loader.get_data(buoy_id_,time-delta,time+delta)
            detection = detector.detect(data_)

            if len(detection) :
                detected_samples.append(detection)
            print(f"{100*count/len(manager):.2f}%",end='\r')
            count+=1

    print(flush=True)
    print(f"{100*len(detected_samples)/len(manager):.2f} of detections in all audios")

if __name__ == "__main__":
    main()
