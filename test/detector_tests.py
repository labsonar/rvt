""" App made to test detectors. """

import os
import shutil
import time
import typing
import argparse
import pandas as pd
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor

from rvt.validate import Validate
from rvt.metric import Metric
from rvt.detector import Detector
from rvt.detectors import energy, zscore, test

DATA_PATH = "./data/RVT/test_files"
data = pd.read_csv("./data/docs/test_artifacts.csv")

FS = 8000

parser = argparse.ArgumentParser(description="App made to test detectors.")

parser.add_argument("-f", "--files", type=int, nargs="*", \
        default=data["Test File"].unique(),
        help="Files to be analysed. Defaults to all.")

parser.add_argument("-m", "--metrics", type=int, nargs="*", \
        default=[i for i in range(len(Metric))],
        help="Metrics to be analysed:\
            0 - Detection Probability\
            1 - False Alarm Rate\
            2 - False Discovery Rate\
            3 - Precision\
            Defaults to all.")

parser.add_argument("-d", "--detector", type=int, default=0,
        help="Detector to be analysed:\
            0 - Energy Threshold Detector\
            1 - Zscore Detector\
            Defaults to all.")

parser.add_argument("-p", "--params", type=float, nargs='*',
                    help="Z-Score Detector Parameters: window_size step threshold.")

args = parser.parse_args()

detector_map = {
    0: (energy.EnergyThresholdDetector, energy.create_energy_config),
    1: (zscore.ZScoreDetector, zscore.create_zscore_config)
}

detector_class, config_creator = detector_map[args.detector]
config = config_creator(args.params)
detector = detector_class(config)

params_dict = {
    config.params_name_list[i]: args.params[i] for i in range(len(args.params))
}

detectores = [detector]
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
    for offset in data[data["Test File"] == file]["Offset"]:
        delta = pd.Timedelta(offset).total_seconds()
        expected.append(int(delta*fs))

    for jndex, detector in enumerate(detectores):
        print(f"\tTesting {detector.name}", end=": ", flush=True)
        # 0.03 - tempo equivalente a 50 jardas
        detector_start = time.time()
        cm = detector.evaluate(input_data, expected, 0.03*fs)
        print(f"{time.time()-detector_start :.1f} seconds")

        validation.accumulate(detector.name, file, cm)

    print(f"Finished reading {file}.wav in {time.time()-start :.1f} seconds")

# TODO usar o ThreadPoolExecutor esta causando problemas pois cada thread acessa a mesma instancia do validate
# def process_file(file: int):
#     """Função para processar um único arquivo."""
#     start = time.time()
#     filename = os.path.join(DATA_PATH, f"{file}.wav")
#     print(f"\nReading {file}.wav:")
#     fs, input_data = wavfile.read(filename)

#     expected = []
#     for offset in data[data["Test File"] == file]["Offset"]:
#         delta = pd.Timedelta(offset).total_seconds()
#         expected.append(int(delta * fs))

#     for jndex, detector in enumerate(detectores):
#         print(f"\tTesting {detector.name}", end=": ", flush=True)
#         detector_start = time.time()
#         cm = detector.evaluate(input_data, expected, 0.03 * fs)
#         print(f"{time.time() - detector_start :.1f} seconds")
#         validation.accumulate(detector.name, file, cm)

#     print(f"Finished reading {file}.wav in {time.time() - start :.1f} seconds")


# # Paralelização usando ThreadPoolExecutor
# with ThreadPoolExecutor() as executor:
#     executor.map(process_file, args.files)

for detector in detectores:
    validation.confusion_matrix(detector.name)

print(validation.build_table(metrics_list, params_dict, detector.name.split()[0]))
