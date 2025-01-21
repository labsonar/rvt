""" Module providing High Pass Filter. """

import typing
import numpy as np

from rvt.preprocessing import PreProcessor

class HighPass(PreProcessor):

    def __init__(self, cut_freq: int = 2000):
        self.cut_freq = cut_freq

    def process(self, input_data: np.ndarray, fs: int) -> typing.Tuple[np.ndarray, int]:
        """ Transform input data to Filter or Normalize data.

        Args:
            input_data (np.ndarray): Audio data.
            fs (int: Audio sampling rate.

        Returns:
            typing.Tuple[np.ndarray, int]: Transformed Audio and Sampling rate.
        """

        data = np.fft.rfft(input_data)
        data = data[self.cut_freq:]
        return np.fft.ifft(data), fs
