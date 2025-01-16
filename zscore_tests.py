""" App made to test detectors. """

import os
import shutil
import time
import typing
import argparse
import pandas as pd
from scipy.io import wavfile

from src.validate import Validate
from src.metric import Metric
from src.detector import Detector
from src.detectors import energy, zscore, test

DATA_PATH = "data/RVT/test_files"
data = pd.read_csv("data/docs/test_files.csv")

FS = 8000

parser = argparse.ArgumentParser(description="App made to test detectors.")

parser.add_argument("-f", "--files", type=int, nargs="*", \
        default=data["Test File ID"].unique(),
        help="Files to be analysed. Defaults to all.")

parser.add_argument("-m", "--metrics", type=int, nargs="*", \
        default=[i for i in range(len(Metric))],
        help="Metrics to be analysed:\
            0 - Detection Probability\
            1 - False Alarm Rate\
            2 - False Discovery Rate\
            3 - Precision\
            Defaults to all.")

parser.add_argument("-d", "--detector", type=int, nargs="*", \
        default=[i for i in range(2)], \
        help="Detectors to be analysed:\
            0 - Energy Threshold Detector\
            1 - Zscore Detector\
            Defaults to all.")

parser.add_argument("-z", "--zscore_params", type=float, nargs=3,
                    default=[20, 4, 2.5],
                    help="Z-Score Detector Parameters: window_size step threshold.")

args = parser.parse_args()

detectores: typing.List[Detector] = [
    energy.EnergyThresholdDetector(),
    zscore.ZScoreDetector(int(args.zscore_params[0]),
                          int(args.zscore_params[1]), args.zscore_params[2])
]

params_dict = {
    "window_size": int(args.zscore_params[0]),
    "step": int(args.zscore_params[1]),
    "threshold": float(args.zscore_params[2])
}

detectores = [detectores[i] for i in args.detector]
metrics_list = [Metric(i) for i in args.metrics]

# if os.path.exists("Result"):
#     shutil.rmtree("Result")

if not os.path.exists("Result"):
    os.makedirs("Result")

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
        print(f"\tTesting {jndex+1}° detector - {detector.name}", end=": ", flush=True)
        # 0.03 - tempo equivalente a 50 jardas
        detector_start = time.time()
        cm = detector.evaluate(input_data, expected, 0.03*fs)
        print(f"{time.time()-detector_start :.1f} seconds")

        validation.accumulate(detector.name, file, cm)

    print(f"Finished reading {file}.wav in {time.time()-start :.1f} seconds")

for detector in detectores:
    validation.confusion_matrix(detector.name)

print(validation.build_table(metrics_list, params_dict))
