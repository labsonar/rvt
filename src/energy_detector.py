""" Module providing energy threshold detector. """
import typing
import numpy as np
from lps_sp.signal import Normalization

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

        self.name = f"Energy Threshold Detector {self.__threshold} - {self.__mean_energy_window_size} - {self.__instant_window_size} - {self.__scaler.name}"

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