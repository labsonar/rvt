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

    def evaluate(self,
                input_data: np.array,
                expected_detections: typing.List[int],
                tolerance: float) -> np.array:
        """ Evaluate detect.

        Args:
            input_data (np.array): Data vector
            expected_detections (typing.List[int]): \
                                Samples expected to be detected
            tolerance (int): percentage offset of size of total samples \
                                to consider samples close enough for detection

        Returns:
            np.array: [True_positive, False Negative;
                        False positive, True Negative]
        """

        confusion_matrix = [[0,0],
                            [0,0]]

        result = self.detect(input_data)
        detections: list = result[0]
        size: int = result[1]

        #print(detections, expected_detections)

        for expected_detection in expected_detections :

            closest_detection_index = bisect_left(detections, expected_detection)

            if closest_detection_index == len(detections):
                # Detection is not valid (True negative)
                confusion_matrix[1][0] += 1
                continue

            closest_detection = detections[closest_detection_index]

            if abs(closest_detection - expected_detection) <= tolerance*size:
                # Detection is valid (True positive)
                # expected_detections.pop(index)
                confusion_matrix[0][0] += 1
            else:
                # Detection is not valid (False Positive)
                confusion_matrix[1][0] += 1

        for detection in detections:
            # Remaining detections (should not be detected)

            closest_expected_detection_index = bisect_left(expected_detections, detection)

            if closest_expected_detection_index == len(expected_detections):
                # Detection is not valid (True negative)
                confusion_matrix[1][1] += 1
                continue

            closest_expected_detection = expected_detections[closest_expected_detection_index]

            if abs(closest_expected_detection - detection) <= tolerance*size:
                # Detection is valid (False negative) (expected to be impossible)
                # detections.pop(index)
                confusion_matrix[0][1] += 1
            else:
                # Detection is not valid (True negative)
                confusion_matrix[1][1] += 1

        # TODO The solution is valid? Not shure if this works in the correct way.
        return confusion_matrix
