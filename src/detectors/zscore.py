import typing
import numpy as np
from src.detector import Detector

class ZScoreDetector(Detector):
    """
    Z-score based anomaly detector inheriting from the abstract Detector class.
    Detects anomalies in time series data using Z-scores.
    """

    def __init__(self, estimation_window_size: int, step: int = 1):
        """
        Parameters:
        - estimation_window_size (int): The size of the moving window to calculate mean and std deviation.
        - step (int, optional): The step size between samples for detection. Defaults to 1.
        """
        self.estimation_window_size = estimation_window_size
        self.step = step

    def detect(self, input_data: np.array, threshold: float = 3, board_only: bool = True) -> typing.Tuple[typing.List[int], int]:
        """
        Perform Z-score based anomaly detector on the given data.

        Parameters:
        - input_data (np.array): The input data array for detection.
        - threshold (float, optional): The Z-score threshold to detect anomalies (default is 3.0).
        - board_only (bool, optional): The detect anomalies are identified only when transitioning
            from non-anomaly to anomaly. The consecutive anomalies are discarted (default is True).

        Returns:
        - np.array: An array of indices (center of the analysis window) where anomalies are
            detected.
        """
        anomalies = []
        for i in range(self.estimation_window_size + self.step,
                    len(input_data),
                    self.step):

            start_index = i - self.estimation_window_size - self.step

            estimation_window = input_data[start_index:start_index + self.estimation_window_size]
            mean = np.mean(estimation_window)
            std = np.std(estimation_window)

            sample_window = input_data[i - self.step:i]
            z_score = np.abs((np.median(sample_window) - mean) / std)

            if z_score > threshold:
                anomalies.append(i - self.step)

        if board_only:
            if len(anomalies) > 1:
                diffs = np.diff(anomalies)
                to_keep = np.insert(diffs > self.step, 0, True)
                anomalies = anomalies[to_keep]

        return anomalies, len(input_data) // self.step
    

# Example usage:
if __name__ == "__main__":

    data = np.random.normal(0, 1, 1000)
    data[300:310] += 10

    detector = ZScoreDetector(estimation_window_size=50, step=10)

    detections = detector.detect(data)
    print("Detections:", detections)

    expected = [305]
    tolerance = 5
    confusion_matrix = detector.evaluate(data, expected, tolerance)
    print("Confusion Matrix:", confusion_matrix)
