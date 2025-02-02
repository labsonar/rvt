""" Pipeline Module
"""
import abc
import typing
import numpy as np

import bisect
import threading

import lps_rvt.dataloader as rvt

class PreProcessor(abc.ABC):
    @abc.abstractmethod
    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        pass

class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, input_data: np.ndarray, samples_to_check: typing.List[int]) -> typing.List[int]:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def evaluate(expected_detections: typing.List[int],
                 expected_rebounds: typing.List[int],
                 samples_to_check: typing.List[int],
                 detect_samples: typing.List[int],
                 tolerance_before: int,
                 tolerance_after: int) -> np.ndarray:

        expected_detections = set(expected_detections)
        expected_rebounds = set(expected_rebounds)
        detect_samples = set(detect_samples)

        detection_ranges = []
        for d in expected_detections:
            detection_ranges.append(set(range(d - tolerance_before, d + tolerance_after + 1)))

        tp = 0
        undetected_expected = detect_samples.copy()

        for expected_range in detection_ranges:
            detected = any(d in expected_range for d in detect_samples)
            if detected:
                tp += 1
                undetected_expected -= expected_range

        rebound_ranges = []
        for d in expected_rebounds:
            rebound_ranges.append(set(range(d - tolerance_before, d + tolerance_after + 1)))

        for rebound_range in rebound_ranges:
            rebound_detected = any(d in rebound_range for d in undetected_expected)
            if rebound_detected:
                undetected_expected -= rebound_range

        fp = len(undetected_expected)

        fn = len(expected_detections) - tp

        tn = len(samples_to_check) - tp - fp - fn

        confusion_matrix = [[tn,fp],
                            [fn,tp]]

        return np.array(confusion_matrix, dtype=np.int32)


class ProcessingPipeline:
    def __init__(self, preprocessors: typing.List[PreProcessor], detectors: typing.List[Detector]):
        self.preprocessors = preprocessors
        self.detectors = detectors
        self.loader = rvt.DataLoader()
        self.margin = 2000
        self.sample_step = 80
        self.tolerance_before=self.sample_step
        self.tolerance_after=self.sample_step

    def process_file(self, file_id: int, result: dict):
        fs, data = self.loader.get_data(file_id)
        expected_detections, expected_rebound = self.loader.get_critical_points(file_id, fs)

        for preprocessor in self.preprocessors:
            fs, data = preprocessor.process(fs, data)

        input_size = len(data)

        samples_to_check = list(range(self.margin, input_size - self.margin, self.sample_step))
        current_samples_to_check = samples_to_check.copy()
        detector_results = {}

        for detector in self.detectors:
            detections = detector.detect(data, current_samples_to_check)
            evaluation = detector.evaluate(expected_detections, samples_to_check, detections,
                                           tolerance_before=self.tolerance_before,
                                           tolerance_after=self.tolerance_after)
            detector_results[str(detector)] = {
                "detections": detections,
                "evaluation": evaluation
            }
            current_samples_to_check = detections

        result[file_id] = detector_results

    def apply(self, files: typing.List[int]) -> dict:
        result = {}
        threads = []

        for file_id in files:
            thread = threading.Thread(target=self.process_file, args=(file_id, result))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return result
