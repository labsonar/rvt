import os
import numpy
from datetime import datetime, timedelta, time
import wave
import pandas as pd
from scipy.io import wavfile

from artifact import ArtifactManager
from loader import DataLoader
from energy_detector import EnergyThresholdDetector
from zscore_detector import ZScoreDetector
from lps_sp.signal import Normalization
import metrics

validate = metrics.Validate()

metrics_list = [metrics.Metric.DETECTION_PROBABILITY,
                metrics.Metric.FALSE_ALARM_RATE,
                metrics.Metric.FALSE_DISCOVERY_RATE]

scalar = Normalization(1)

detectores = {
    "energy": EnergyThresholdDetector(20,1600,100,scalar),
    "zscore": ZScoreDetector(1000, 500)
}

data_path = "Data/RVT/test_data"
data = pd.read_csv("rvt/data/docs/dados_teste.csv")

files = data["Files"].unique()
print(files)

for file in files:
    
    filename = os.path.join(data_path, f"{file}.wav")
    fs, input = wavfile.read(filename)
    # print(fs, input)
    
    gabarito = []

    for offset in data[data["Files"] == file]["Offset"]:

        delta = pd.Timedelta(offset).total_seconds()
        gabarito.append(int(delta*fs))

    for label, detector in detectores.items():

        print(label)
        # 0.03 - tempo equivalente a 50 jardas
        cm = detector.evaluate(input, gabarito, 0.03*fs)

        validate.accumulate(label, cm)


print(validate.build_table(metrics_list))

