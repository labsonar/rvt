""" App made to test detectors. """

import os
import shutil
import time
import typing
import argparse
import pandas as pd
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from rvt.validate import Validate
from rvt.metric import Metric
from rvt.detector import Detector
from rvt import test_loader
from rvt.detectors import energy, zscore, hypothesis, test
from rvt.preprocessing import PreProcessor, ProcessorPipeline
from rvt.preprocessors import high_pass

data = pd.read_csv("./data/docs/test_artifacts.csv")

# TODO generalizar frequencia de corte do filtro
pre_processors: typing.List[PreProcessor] = [
    high_pass.HighPass(1000)
]

FS = 8000

parser = argparse.ArgumentParser(description="App made to test detectors.")

parser.add_argument("-f", "--files", type=int, nargs="*",
        default=data["Test File"].unique(),
        help="Files to be analysed. Defaults to all.")

parser.add_argument("-a", "--ammo_types", type=str, nargs="*",
                    default=['EX-SUP', 'HE3m', 'GAE'],
                    help="Ammo types list")

parser.add_argument("-b", "--buoy_ids", type=int, nargs='*',
                    default=[1, 2, 3, 4, 5],
                    help="Buoy IDs list")

parser.add_argument("-m", "--metrics", type=int, nargs="*",
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
            2 - Hypothesis Detector\
            Defaults to all.")

parser.add_argument("-p", "--preprocessor", type=int, nargs="*",
        default=None,
        help="Pre Processors to be used in order:\
            0 - High Pass Filter")

parser.add_argument("--params", type=float, nargs='*',
                    help="Detector Parameters List: window_size step threshold.")

parser.add_argument("-t", "--test", action='store_true',
                    help='Choose test or validation set')

args = parser.parse_args()

detector_map = {
    0: (energy.EnergyThresholdDetector, energy.create_energy_config),
    1: (zscore.ZScoreDetector, zscore.create_zscore_config),
    2: (hypothesis.HypothesisDetector, hypothesis.create_hypothesis_config)
}

detector_class, config_creator = detector_map[args.detector]
config = config_creator(args.params)
detector = detector_class(config)

params_dict = {
    config.params_name_list[i]: args.params[i] for i in range(len(args.params))
}

detectores = [detector]

if args.preprocessor:
    pre_processing = ProcessorPipeline([pre_processors[i] for i in args.preprocessor])

metrics_list = [Metric(i) for i in args.metrics]

# if os.path.exists("Result"):
#     shutil.rmtree("Result")

if not os.path.exists("Result"):
    os.makedirs("Result")

file_loader = test_loader.DataLoader()

file_set = {file_id for ammo_type in args.ammo_types 
                     for buoy_id in args.buoy_ids
                     for file_id in file_loader.getID(ammo_type, buoy_id, args.test)}

file_list = [file for file in args.files if file in file_set]

print(f"Selected Files: {file_list}")

validation = Validate(file_list, detectores, "Result")

def process_file(file: int):
    # try:
    cm_dict = {}
    start = time.time()
    fs, input_data = file_loader.getData(file)

    if args.preprocessor:
        input_data, fs = pre_processing.process(input_data, fs)

    expected = []
    for offset in data[data["Test File"] == file]["Offset"]:
        delta = pd.Timedelta(offset).total_seconds()
        expected.append(int(delta * fs))
        
    # print(f"Expected Detections: {expected}")

    for jndex, detector in enumerate(detectores):
        # print(f"\tTesting {detector.name}", end=": ", flush=True)
        detector_start = time.time()
        cm = detector.evaluate(input_data, expected, 2000)
        cm_dict[file] = cm
        print(f"Finished processing file {file} in {time.time() - detector_start :.1f} seconds")

    # print(f"Finished reading {file}.wav in {time.time() - start :.1f} seconds")

    return cm_dict

    # except Exception as e:
    #     print(f"Erro ao processar o arquivo {file}: {e}")
    #     return None

with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_file, file_list))

cm_dict = {}
for result in results:
    if result:
        cm_dict.update(result)

for file in file_list:
    validation.accumulate(detector.name, file, cm_dict[file])

for detector in detectores:
    validation.confusion_matrix(detector.name)

print(validation.build_table(metrics_list, params_dict, detector.name.split()[0]))
