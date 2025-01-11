import os
import shutil
import typing
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint, choice

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
                metrics.Metric.PRECISION]

detectores: typing.List[Detector] = []

# FS = 8000
# for i in range(25):
#     desvio_media = randint(100,5000)/100
#     janela_grande = round(randint(30,100)/100*FS)
#     janela_pequena = round(randint(1000,3000)/100000*FS)
#     scaler = Normalization(choice([1,4]))
    
#     detectores.append(EnergyThresholdDetector(desvio_media, janela_grande, janela_pequena, scaler))

detectores: typing.List[Detector] = [
    EnergyThresholdDetector(35, 4800, 80, Normalization(1)),
    # ZScoreDetector(1000, 500)
]

data_path = "data/RVT/test_files"
data = pd.read_csv("data/docs/test_files.csv")

files = [1]
#files = data["Test File ID"].unique()

for file in files:
    
    filename = os.path.join(data_path, f"{file}.wav")
    print(filename)
    fs, input = wavfile.read(filename)
    print(fs)
    
    gabarito = []

    for offset in data[data["Test File ID"] == file]["Offset"]:

        delta = pd.Timedelta(offset).total_seconds()
        gabarito.append(int(delta*fs))

    for jndex, detector in enumerate(detectores):

        print(jndex, detector.name)
        # 0.03 - tempo equivalente a 50 jardas
        cm = detector.evaluate(input, gabarito, 0.03*fs)

        validate.accumulate(detector.name, cm)

for detector in detectores:
    validate.confusion_matrix(detector.name, "Result")

validate.build_table(metrics_list).to_csv("Result/table.csv")

