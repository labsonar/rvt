""" Module providing abstract pre processing unit. """
import abc
import typing
import numpy as np

class PreProcessor(abc.ABC):

    @abc.abstractmethod
    def process(self, input_data: np.ndarray, fs: int) -> typing.Tuple[np.ndarray, int]:
        """ Transform input data to Filter or Normalize data.

        Args:
            input_data (np.ndarray): Audio data.
            fs (int: Audio sampling rate.

        Returns:
            typing.Tuple[np.ndarray, int]: Transformed Audio and Sampling rate.
        """

class ProcessorPipeline(PreProcessor):

    def __init__(self, pre_processors: typing.List[PreProcessor]):
        self.pre_processors = pre_processors

    def process(self, input_data: np.ndarray, fs: int) -> typing.Tuple[np.ndarray, int]:
        """ Transform input data to Filter or Normalize data.

        Args:
            input_data (np.ndarray): Audio data.
            fs (int: Audio sampling rate.

        Returns:
            typing.Tuple[np.ndarray, int]: Transformed Audio and Sampling rate.
        """

        for processor in self.pre_processors:
            input_data, fs = processor.process(input_data, fs)

        return input_data, fs