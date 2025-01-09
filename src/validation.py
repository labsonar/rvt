import os
import shutil
import typing
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns

from detector import Detector
from energy_detector import EnergyThresholdDetector
from zscore_detector import ZScoreDetector
from test_detector import TestDetector
from lps_sp.signal import Normalization
import metrics

if os.path.exists("Result"):
    shutil.rmtree("Result")
os.mkdir("Result")

validate = metrics.Validate()

metrics_list = [metrics.Metric.DETECTION_PROBABILITY,
                metrics.Metric.FALSE_ALARM_RATE,
                metrics.Metric.FALSE_DISCOVERY_RATE]

scalar = Normalization(1)

detectores: typing.Dict[str, Detector] = {
    "test True": TestDetector(True),
    "test False": TestDetector(False),
    "energy 2.5 1600 10": EnergyThresholdDetector(2.5, 1600, 10, scalar),
    "energy 0.001 2 1": EnergyThresholdDetector(0.001, 2, 1, scalar),
    "energy 20 1600 160": EnergyThresholdDetector(20, 1600, 160, scalar),
    # "zscore": ZScoreDetector(1000, 500)
}

data_path = "data/RVT/test_files"
data = pd.read_csv("data/docs/test_files.csv")

files = data["Test File ID"].unique()

for file in files:
    
    filename = os.path.join(data_path, f"{file}.wav")
    print(filename)
    fs, input = wavfile.read(filename)
    
    gabarito = []

    for offset in data[data["Test File ID"] == file]["Offset"]:

        delta = pd.Timedelta(offset).total_seconds()
        gabarito.append(int(delta*fs))

    for label, detector in detectores.items():

        print(label)
        # 0.03 - tempo equivalente a 50 jardas
        cm = detector.evaluate(input, gabarito, 0.03*fs)
        
        sns.heatmap(cm, annot=True, cmap="magma", linewidths=0.5, cbar=False)
        plt.title(f"{label}_{file}")
        plt.savefig(f"Result/{label}_{file}.png")

        plt.close('all')
        sns.reset_defaults()

        validate.accumulate(label, cm)


print(validate.build_table(metrics_list))

