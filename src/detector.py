""" Module providing abstract detector. """
import typing
import abc
from bisect import bisect_left
import numpy as np

class Detector(abc.ABC):
    """ Abstract detector for acoustical events. """

    @abc.abstractmethod
    def detect(self, input_data: np.array) -> typing.Tuple[typing.List[int], int]:
        """
        Args:
            input_data (np.array): Data vector.

        Returns:
            typing.Tuple[typing.List[int], int]: Detected samples and number of windows.
        """

    def evaluate(self, input_data: np.array, expected_detections: typing.List[int],
                tolerance: int) -> np.array:
        """ Evaluate detect.

        Args:
            input_data (np.array): Data vector.
            expected_detections (typing.List[int]): \
                                Samples expected to be detected.
            tolerance (int): Offset to consider samples close enough for detection.

        Returns:
            np.array: [True_positive, False Negative;
                        False positive, True Negative]
        """

        confusion_matrix = [[0,0],
                            [0,0]]

        result = self.detect(input_data)
        detections: list = result[0]
        size: int = result[1]

        to_erase_expected_detection: list = []
        to_erase_detection: list = []

        for index, expected_detection in enumerate(expected_detections):

            closest_detection_index = bisect_left(detections, expected_detection)

            if closest_detection_index == len(detections):
                continue

            closest_detection = detections[closest_detection_index]

            if abs(closest_detection - expected_detection) <= tolerance:
                # Valid detection found (True positive)
                detections.pop(closest_detection_index)
                to_erase_expected_detection.append(index)
                confusion_matrix[0][0] += 1

        new_expected_detections: list = []
        for index, element in enumerate(expected_detections):
            if not index in to_erase_expected_detection:
                new_expected_detections.append(element)

        expected_detections = new_expected_detections

        for index, detection in enumerate(detections):

            closest_expected_detection_index = bisect_left(expected_detections, detection)

            if closest_expected_detection_index == len(expected_detections):
                continue

            closest_expected_detection = expected_detections[closest_expected_detection_index]

            if abs(closest_expected_detection - detection) <= tolerance:
                # Detection expected (True positive) Should be impossible
                expected_detections.pop(closest_expected_detection_index)
                to_erase_detection.append(index)
                confusion_matrix[0][0] += 1

        new_detections: list = []
        for index, element in enumerate(detections):
            if not index in to_erase_detection:
                new_detections.append(element)

        detections = new_detections

        # Remaining expected detections not found are False Negatives
        confusion_matrix[0][1] += len(expected_detections)

        # Remaining detections are False Positives
        confusion_matrix[1][0] += len(detections)

        # All of the remaining samples analysed are True Negatives
        confusion_matrix[1][1] += size - confusion_matrix[0][0]\
                                - confusion_matrix[0][1] - confusion_matrix[1][0]

        # TODO The solution is valid? Not shure if this works in the correct way.
        return confusion_matrix
