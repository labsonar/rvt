""" Module providing energy threshold detector. """
import typing
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt # only for confusion matrix
import seaborn as sns

from lps_sp.signal import Normalization

from artifact import ArtifactManager
from loader import DataLoader
from detector import Detector

class EnergyThresholdDetector(Detector):
    """ Class representing an energy threshold detector. """

    def __init__(self, threshold: float, mean_energy_window_size: int, instant_window_size: int, \
            scaler: typing.Optional[int] = 4):
        self.__threshold: float = threshold
        self.__mean_energy_window_size: int = mean_energy_window_size
        self.__instant_window_size: int = instant_window_size
        self.__scaler: Normalization = Normalization(scaler)

        if self.__instant_window_size >= self.__mean_energy_window_size:
            # TODO See if this print is okay
            raise ValueError(f"Instant window {self.__instant_window_size} \
                            greater or equal mean energy window {self.__mean_energy_window_size}")

    def detect(self, input_data: np.array) -> typing.Tuple[typing.List[int], int]:
        """
        Args:
            input_data (np.array): Data vector.

        Returns:
            typing.Tuple[typing.List[int], int]: Detected samples and number of windows.
        """

        if self.__mean_energy_window_size > len(input_data):
            # TODO See if this print is okay
            raise ValueError(f"Window Size ({self.__mean_energy_window_size}) > \
                            data ({len(input_data)}).")

        if self.__scaler:
            # TODO Should apply **2 to input_data before or after normalization?
            input_data = self.__scaler.apply(input_data)**2

        detected_samples = []

        # First window
        mean_energy_sum: float = np.sum(input_data[:self.__mean_energy_window_size])

        instant_energy_sum: float = np.sum(input_data\
                                    [self.__mean_energy_window_size-self.__instant_window_size:\
                                    self.__mean_energy_window_size])

        # For next windows
        detection_occuring = False
        for index in range(len(input_data) - self.__mean_energy_window_size):

            l_mean = index
            r_mean = index + self.__mean_energy_window_size

            r_instant = r_mean
            l_instant = r_instant - self.__instant_window_size

            instant_energy_sum -= input_data[l_instant]
            instant_energy_sum += input_data[r_instant]

            mean_energy = mean_energy_sum/self.__mean_energy_window_size
            instant_energy = instant_energy_sum/self.__instant_window_size

            if instant_energy > self.__threshold * mean_energy:

                if not detection_occuring:
                    detection_occuring = True
                    detected_samples.append(r_instant)

            elif detection_occuring:
                detection_occuring = False

            mean_energy_sum -= input_data[l_mean]
            mean_energy_sum += input_data[r_mean]

        return detected_samples, len(input_data)-self.__mean_energy_window_size

def main():
    """ Main for debuging. """
    scalar = Normalization(1)

    detector = EnergyThresholdDetector(20,1600,100,scalar) # TODO This tuning seens to be off
    manager = ArtifactManager("data/docs/development.csv")
    loader = DataLoader()

    delta = timedelta(seconds=20)

    detected_samples = []
    count = 1
    for id_artifact in manager:
        for buoy_id_, time in manager[id_artifact]:

            fr , data_ = loader.get_data(buoy_id_,time-delta,time+delta)
            detections, samples = detector.detect(data_)

            if len(detections)!=0 :
                detected_samples.append(detections)

            print(f"{len(detections)} detections in sample")
            print(f"{round(100*count/len(manager)):.2f}%",end='\r')
            count+=1

    print(flush=True)
    print(f"{round(100*len(detected_samples)/len(manager)):.2f}% of detections in all audios")

def evaluate_test():
    "evaluate test for debugging abstract detector."
    scalar = Normalization(1)

    detector = EnergyThresholdDetector(20,1600,100,scalar) # TODO This tuning seens to be off

    manager = ArtifactManager("data/docs/development.csv")
    loader = DataLoader()

    start = datetime(2023, 9, 12, 17, 35)
    end = datetime(2023, 9, 12, 18, 25)
    buoy = 2

    fs, input_data = loader.get_data(buoy, start, end)

    expected_detections_datetimes = manager.artifact_amount_by_time(buoy, start, end)

    expected_detections = loader.get_index(start, fs, expected_detections_datetimes)

    matrix = detector.evaluate(input_data,expected_detections,0.1*len(input_data))

    plt.figure(figsize=(10,10))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm", cbar=False,
                    annot_kws={"size": 35, "color": "black"})

    # Show the plot
    #plt.tight_layout()
    plt.savefig("data/Analysis/img.png")

if __name__ == "__main__":
    evaluate_test()
