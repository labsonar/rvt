""" App made to test detectors. """

import os
import shutil
import time
import typing
import argparse
import pandas as pd
from scipy.io import wavfile

#from lps_sp.signal import Normalization

from src.validate import Validate
from src.metric import Metric
from src.detector import Detector
from src.detectors import energy, zscore, test

DATA_PATH = "data/RVT/test_files"
data = pd.read_csv("data/docs/test_files.csv")

parser = argparse.ArgumentParser(description=" App made to test detectors. ")

parser.add_argument("-f", "--files", type=int, nargs="*", \
    help="Files to be analysed", default=data["Test File ID"].unique())

parser.add_argument("-m", "--metrics", type=int, nargs="*", \
    help="Metrics to be analysed", default=[i for i in range(len(Metric))])

args = parser.parse_args()

metrics_list = [Metric(i) for i in args.metrics]

if os.path.exists("Result"):
    shutil.rmtree("Result")
os.mkdir("Result")

detectores: typing.List[Detector] = [
    energy.EnergyThresholdDetector(),
    #zscore.ZScoreDetector(1000, 500),
    test.TestDetector()
]

validation = Validate(args.files, detectores, "Result")

for file in args.files:

    start = time.time()

    filename = os.path.join(DATA_PATH, f"{file}.wav")
    print(f"\nReading {file}.wav:")
    fs, input_data = wavfile.read(filename)

    expected = []
    for offset in data[data["Test File ID"] == file]["Offset"]:
        delta = pd.Timedelta(offset).total_seconds()
        expected.append(int(delta*fs))

    for jndex, detector in enumerate(detectores):
        print(f"Testing {jndex+1}° detector - {detector.name}", end=": ", flush=True)
        # 0.03 - tempo equivalente a 50 jardas
        detector_start = time.time()
        cm = detector.evaluate(input_data, expected, 0.03*fs)
        print(f"{time.time()-detector_start :.1f} seconds")

        validation.accumulate(detector.name, file, cm)

    print(f"Finished reading {file}.wav in {time.time()-start :.1f} seconds")

for detector in detectores:
    validation.confusion_matrix(detector.name)

print(validation.build_table(metrics_list))
