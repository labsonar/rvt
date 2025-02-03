import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from scipy.io import wavfile
import scipy.signal as signal
from dataclasses import dataclass, field

import lps_sp.signal as lps_signal
from rvt import detector
from rvt.preprocessors import high_pass
from rvt.detectors import preprocess


@dataclass
class WaveletConfig:
    window_size: int = 2000
    overlap: float = 0.2
    threshold: float = 0.3
    wavelet: str = "db4"
    level: int = 2
    params_name_list: list = field(default_factory=lambda: [
        "Window Size",
        "Overlap",
        "Threshold",
        "Wavelet",
        "Level"
    ])

def create_wavelet_config(params):
    return WaveletConfig(*params) if params else WaveletConfig()

class WaveletDetector(detector.Detector):

    def __init__(self, config: WaveletConfig, scaler: lps_signal.Normalization = lps_signal.Normalization(1)):
        """
        Anomaly Detector using the Discrete Wavelet Transform (DWT).
        """
        self.window_size = int(config.window_size)
        self.overlap = float(config.overlap)
        self.threshold = float(config.threshold)
        self.wavelet = str(config.wavelet)
        self.level = int(config.level)
        self.name = f"Wavelet Detector - {self.window_size} - {self.overlap} - {self.threshold} - {self.wavelet} - {self.level}"
        self.scaler = scaler

    def detect(self, input_data):
        """
        Detects anomalies using the DWT and comparing detail coefficients' energy with given threshold.
        """

        anomalies = []
        step = int(self.window_size * (1 - self.overlap))

        input_data = self.scaler.apply(input_data)

        for idx, window in enumerate(self.sliding_window(input_data, self.window_size, self.overlap)):
            coeffs = pywt.wavedec(window, self.wavelet, level=self.level)
            
            detail_coeffs = np.concatenate(coeffs[1:])  
            energy = np.sum(detail_coeffs ** 2)

            # print(f"Energy: {energy}")

            if energy > self.threshold:
                anomalies.append(idx * step)

        # print(f"Detected Anomalies: {anomalies}")

        return anomalies, len(input_data)

    def sliding_window(self, input_data, window_size, overlap):
        """
        Divides the signal in sliding windows
        """
        step = int(window_size * (1 - overlap))
        for start in range(0, len(input_data) - window_size + 1, step):
            yield input_data[start:start + window_size]


def plot_results(input_data, fs, anomalies, window_size, expected_detections):
    """
    Plots audio waveform, highlighting detected anomalies and ground truth samples.
    """
    samples = np.arange(len(input_data))
    plt.figure(figsize=(12, 6))
    plt.plot(samples, input_data, label="Audio Waveform")

    anomaly_label_shown = False
    for start_sample in anomalies:
        plt.axvspan(
            start_sample, start_sample + window_size,
            color='red', alpha=0.3,
            label='Detected Anomalies' if not anomaly_label_shown else None
        )
        anomaly_label_shown = True

    plt.scatter(
        expected_detections,
        [input_data[detection] for detection in expected_detections],
        color='green',
        label='Ground Truth',
        zorder=5,
        s=40,
        marker='o'
    )
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(f'Wavelet Detector - File {file}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"wavelet_{file}.png")
    plt.close()


def main(filename, data, window_size=12000, overlap=0.7, threshold=2.0, wavelet="db4", level=4):
    """
    Tests given audio file in the Wavelet Transform Detector
    """
    fs, input_data = wavfile.read(filename)

    expected = []
    for offset in data[data["Test File ID"] == file]["Offset"]:
        delta = pd.Timedelta(offset).total_seconds()
        expected.append(int(delta * fs))
    print(f"Expected Detections: {expected}")

    config = WaveletConfig(window_size, overlap, threshold, wavelet, level)
    detector = WaveletDetector(config)

    anomalies, size = detector.detect(input_data)

    plot_results(input_data, fs, anomalies, window_size, expected)


if __name__ == "__main__":
    DATA_PATH = "./data/RVT/test_files"
    data = pd.read_csv("./data/docs/test_artifacts.csv")

    file = 3
    filename = os.path.join(DATA_PATH, f"{file}.wav")

    print(f"\nReading {file}.wav:")

    main(filename, data, window_size=2000, overlap=0.2, threshold=0.3, wavelet="db4", level=1)
