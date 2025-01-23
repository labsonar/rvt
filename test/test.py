""" App made to test detectors. """

import os
import shutil
import time
import typing
import argparse
import pandas as pd
from scipy.io import wavfile

from rvt.validate import Validate
from rvt.metric import Metric
from rvt.detector import Detector
from rvt.artifact import ArtifactManager
from rvt.loader import DataLoader
from rvt.preprocessing import PreProcessor, ProcessorPipeline
from rvt.detectors import energy, zscore, test
from rvt.preprocessors import high_pass

DATA_PATH = "./data/RVT/test_files"
manager = ArtifactManager("data/docs/test_artifacts.csv")
# loader = DataLoader("") # TODO Use new dataloader

FS = 8000
detectors: typing.List[Detector] = [
    energy.EnergyThresholdDetector(),
    zscore.ZScoreDetector()
]

pre_processors: typing.List[PreProcessor] = [
    high_pass.HighPass(1000)
]

parser = argparse.ArgumentParser(description="App made to test detectors.")

parser.add_argument("-f", "--files", type=int, nargs="*", \
        default=manager.data["Test File"].unique().tolist(),
        help="Files to be analysed. Default to all.")

parser.add_argument("-m", "--metrics", type=int, nargs="*", \
        default=[i for i in range(len(Metric))],
        help="Metrics to be analysed:\
            0 - Detection Probability\
            1 - False Alarm Rate\
            2 - False Discovery Rate\
            3 - Precision\
            Default to all.")

parser.add_argument("-d", "--detector", type=int, nargs="*", \
        default=[i for i in range(len(detectors))], \
        help="Detectors to be analysed:\
            0 - Energy Threshold Detector\
            1 - Zscore Detector\
            Default to all.")

parser.add_argument("-p", "--preprocessor", type=int, nargs="*", \
        default=None, \
        help="Pre Processors to be used in order:\
            0 - High Pass Filter")

args = parser.parse_args()

detectors = [detectors[i] for i in args.detector]

if args.preprocessor:
    pre_processing = ProcessorPipeline([pre_processors[i] for i in args.preprocessor])

metrics_list = [Metric(i) for i in args.metrics]

if os.path.exists("Result"):
    shutil.rmtree("Result")
os.mkdir("Result")

validation = Validate(args.files, detectors, "Result")

for file in args.files:

    start = time.time()

    filename = os.path.join(DATA_PATH, f"{file}.wav")
    print(f"\nReading {file}.wav:")
    fs, input_data = wavfile.read(filename)

    if args.preprocessor:
        input_data, fs = pre_processing.process(input_data, fs)

    expected = []
    for offset in manager.data[manager.data["Test File"] == file]["Offset"]:
        delta = pd.Timedelta(offset).total_seconds()
        expected.append(int(delta*fs))

    for jndex, detector in enumerate(detectors):
        print(f"\tTesting {jndex+1}° detector - {detector.name}", end=": ", flush=True)
        # 0.03 - tempo equivalente a 50 jardas
        detector_start = time.time()
        cm = detector.evaluate(input_data, expected, 0.03*fs)
        print(f"{time.time()-detector_start :.1f} seconds")

        validation.accumulate(detector.name, file, cm)

    print(f"Finished reading {file}.wav in {time.time()-start :.1f} seconds")

for detector in detectors:
    validation.confusion_matrix(detector.name)

print(validation.build_table(metrics_list))
