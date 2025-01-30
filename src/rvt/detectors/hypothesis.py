import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
import scipy.signal as signal
import scipy.stats as stats
from scipy.stats import linregress, ks_2samp
from scipy.io import wavfile

import lps_sp.signal as lps_signal
from rvt import detector

@dataclass
class HypothesisConfig():
    window_size: int = 12000
    overlap: float = 0.7
    threshold: float = 0.2
    params_name_list: list = field(default_factory=lambda: [
        "Window Size",
        "Overlap",
        "Threshold"
    ])

def create_hypothesis_config(params):
    return HypothesisConfig(*params) if params else HypothesisConfig()


class HypothesisDetector(detector.Detector):

    def __init__(self, config: HypothesisConfig = HypothesisConfig(),
                 scaler: lps_signal.Normalization = lps_signal.Normalization(1)):
        self.window_size = int(config.window_size)
        self.overlap = config.overlap
        self.threshold = config.threshold
        self.name = f"Hypothesis Detector - {self.window_size} - {self.overlap} - {self.threshold}"
        self.scaler = scaler

    def detect(self, input_data, fs = 8000):

        anomalies = []
        
        input_data = self.scaler.apply(input_data)
        
        step = int(self.window_size * (1 - self.overlap))
        
        for idx, window in enumerate(self.sliding_window(input_data, self.window_size, self.overlap)):
            freqs, smoothed_psd = self.psd_moving_average(window, fs)
            slope = self.analyze_slope(freqs, smoothed_psd)
            is_anomaly = self.hypothesis_test(slope, self.threshold)
            if is_anomaly:
                anomalies.append(idx * step)
    
        # print(f"Anomalies (samples): {anomalies}")
        
        return anomalies, len(input_data)

    def sliding_window(self, signal, window_size, overlap):
        step = int(window_size * (1 - overlap))
        for start in range(0, len(signal) - window_size + 1, step):
            yield signal[start:start + window_size]

    def psd_moving_average(self, window, fs, moving_average_window=500):
        freqs, psd = signal.welch(window, fs, nperseg=len(window))
        smoothed_psd = np.convolve(psd, np.ones(moving_average_window)/moving_average_window, mode='same')
        return freqs, smoothed_psd

    def analyze_slope(self, freqs, psd):
        log_freqs = np.log(freqs[1:])
        log_psd = np.log(psd[1:])
        slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_psd)
        return slope

    def hypothesis_test(self, slope, threshold):
        return abs(slope) < threshold
    

def plot_results(signal, fs, anomalies, window_size, expected_detections):
    samples = np.arange(len(signal))
    plt.figure(figsize=(12, 6))
    plt.plot(samples, signal, label='Audio Signal')
    
    anomaly_label_shown = False
    for start_sample in anomalies:
        plt.axvspan(start_sample, start_sample + window_size, 
                    color='red', alpha=0.3, 
                    label='Detected Anomalies' if not anomaly_label_shown else None)
        anomaly_label_shown = True
    plt.scatter(
        expected_detections,
        [signal[detection] for detection in expected_detections],
        color='green',
        label='Ground Truth',
        zorder=5,
        s=40,
        marker='o'
    )
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Slope Anomaly Detection - File 1')
    # plt.legend()
    plt.savefig(f"hypo_15_07_02.png")
    plt.close()

def main(filename, data, window_size=2000, overlap=0.7, threshold=0.2):
    fs, input_data = wavfile.read(filename)
    expected = []
    for offset in data[data["Test File"] == file]["Offset"]:
        delta = pd.Timedelta(offset).total_seconds()
        expected.append(int(delta * fs))
    print(f"Expected Detections: {expected}")
    config = HypothesisConfig(window_size, overlap, threshold)
    detector = HypothesisDetector(config)
    anomalies, size = detector.detect(input_data, fs)
    norm = lps_signal.Normalization(1)
    input_data = norm.apply(input_data)
    plot_results(input_data, fs, anomalies, window_size, expected)


if __name__ == '__main__':
    DATA_PATH = "./data/RVT/test_files"
    data = pd.read_csv("./data/docs/test_artifacts.csv")

    file = 23
    filename = os.path.join(DATA_PATH, f"{file}.wav")
    
    print(f"\nReading {file}.wav:")

    main(filename, data, window_size=12000, overlap=0.7, threshold=0.5)