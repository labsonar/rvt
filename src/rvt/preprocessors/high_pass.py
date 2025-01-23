""" Module providing High Pass Filter. """

import typing
import numpy as np
import scipy.signal as signal

from rvt.preprocessing import PreProcessor

class HighPass(PreProcessor):

    def __init__(self, cutoff_freq: int = 2000, order: int = 1):
        """ 
        High Pass Filter

        Args:
            cutoff_freq (int): filter cutoff frequency. Defaults to 2000.
            order (int): order of the filter. Defaults to 1
        """
        
        self.cutoff_freq = cutoff_freq
        self.order = order

    def process(self, input_data: np.ndarray, fs: int) -> typing.Tuple[np.ndarray, int]:
        """ Filters input data.

        Args:
            input_data (np.ndarray): Audio data.
            fs (int): Audio sampling rate.
            order : order of the filter.

        Returns:
            typing.Tuple[np.ndarray, int]: Filtered Audio and Sampling rate.
        """
        
        normalized_cutoff = self.cutoff_freq / (fs / 2.0)
        b, a = signal.butter(self.order, normalized_cutoff, btype='high', analog=False)
        filtered_data = signal.filtfilt(b, a, input_data)
        
        return filtered_data, fs
