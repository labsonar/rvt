import typing
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import lps_sp.signal as lps_signal
from rvt.detector import Detector

@dataclass
class ZScoreConfig():
  """
  Parameters:
  - estimation_window_size (int, optional): The size of the moving window to calculate mean
    and standard deviation. Defaults to 800.
  - step (int, optional): The step size between samples for detection. Defaults to 20.
  - threshold (float, optional): The Z-score threshold to detect anomalies. Default is 3.0.
  - params_name_list (list, optional): List of names for ZScore Detector parameters.
  """
  estimation_window_size: int = 800
  step: int = 20
  threshold: float = 3
  params_name_list: list = [
    "Window Size",
    "Step",
    "Threshold"
  ]
  
def create_zscore_config(params):
    return ZScoreConfig(*params) if params else ZScoreConfig()

class ZScoreDetector(Detector):
    """
    Z-score based anomaly detector inheriting from the abstract Detector class.
    Detects anomalies in time series data using Z-scores.
    """

    def __init__(self, config: ZScoreConfig = ZScoreConfig(),
                 scaler: lps_signal.Normalization = lps_signal.Normalization(1),
                 border_only: bool = True):
        """
        Parameters:
        - config (ZScoreConfig): config dataclass with detector parameters, including
          window size, step size and threshold.
        - scaler (Normalization, optional): Normalization method to apply to input data.
          Default is MIN_MAX_ZERO_CENTERED.
        - border_only (bool, optional): The detect anomalies are identified only when transitioning
          from non-anomaly to anomaly. The consecutive anomalies are discarted. Default is True.
        """
        self.estimation_window_size = int(config.estimation_window_size)
        self.step = int(config.step)
        self.threshold = config.threshold
        self.scaler = scaler
        self.border_only = border_only
        self.name = f"Zscore Detector - {self.estimation_window_size} - {self.step} - {self.threshold} - {self.scaler.name}"

    def detect(self, input_data: np.array) -> typing.Tuple[typing.List[int], int]:
      """
      Perform Z-score based anomaly detector on the given data.

      Parameters:
      - input_data (np.array): The input data array for detection.

      Returns:
      - np.array: An array of indices (center of the analysis window) where anomalies are
          detected.
      """
      anomalies = []
      
      if self.scaler:
          input_data = self.scaler.apply(input_data)
      
      for i in range(self.estimation_window_size + self.step,
                      len(input_data),
                      self.step):

          start_index = i-self.estimation_window_size-self.step

          estimation_window = input_data[start_index:start_index+self.estimation_window_size]
          mean = np.mean(estimation_window)
          std = np.std(estimation_window)

          sample_window = input_data[i-self.step:i]
          # z_scores = np.abs((np.median(sample_window) - mean) / std)
          z_scores = np.abs((np.mean(sample_window) - mean) / std)

          if z_scores > self.threshold:
              anomalies.append(i - self.step)

      anomalies = np.array(anomalies)

      if self.border_only:
          if len(anomalies) > 1:
              diffs = np.diff(anomalies)
              to_keep = np.insert(diffs > self.step, 0, True)
              anomalies = anomalies[to_keep]
              
      # plt.plot(input_data)
      # plt.scatter(anomalies, input_data[anomalies], color='red', marker='o', s=50, label='Anomalias', zorder=5)
      # plt.savefig("teste.png")

      return anomalies.tolist(), len(input_data) // self.step
    
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
