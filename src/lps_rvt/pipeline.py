""" Pipeline Module
"""
import abc
import typing
import numpy as np

import threading
import plotly.graph_objs as go
import scipy.signal as scipy
import streamlit as st

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

class Result:
    def __init__(self, file_id: int):
        self.file_id = file_id
        self.fs = None
        self.expected_detections = []
        self.expected_rebounds = []
        self.processed_signal = None
        self.detections = {}
        self.evaluations = {}

    def st_show_final_plot(self):
        """Generates a Plotly plot with the data"""

        num_samples = 400000
        data_resampled = scipy.resample(self.processed_signal, num_samples)

        original_samples = len(self.processed_signal)
        resampling_factor = original_samples / num_samples
        new_fs = self.fs / resampling_factor

        time_axis = [i / new_fs for i in range(len(data_resampled))]

        trace_signal = go.Scatter(x=time_axis, y=data_resampled, mode='lines', name='Signal Data')

        shapes = []
        for d in self.expected_detections:
            new_d = int(d/resampling_factor)
            shapes.append(
                dict(
                    type="line",
                    x0=time_axis[new_d],
                    y0=min(data_resampled),
                    x1=time_axis[new_d],
                    y1=max(data_resampled),
                    line=dict(color="darkgreen", width=1, dash="dot")
                )
            )

        layout = go.Layout(
            title=f"Arquivo {self.file_id}",
            xaxis=dict(title="Time (seconds)"),
            yaxis=dict(title="Amplitude"),
            showlegend=True,
            shapes=shapes
        )

        fig = go.Figure(data=[trace_signal], layout=layout)
        st.plotly_chart(fig)


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
        result[file_id] = Result(file_id)

        fs, data = self.loader.get_data(file_id)
        result[file_id].expected_detections, result[file_id].expected_rebound = \
                    self.loader.get_critical_points(file_id, fs)

        for preprocessor in self.preprocessors:
            fs, data = preprocessor.process(fs, data)

        result[file_id].fs = fs
        result[file_id].processed_signal = data

        input_size = len(data)

        samples_to_check = list(range(self.margin, input_size - self.margin, self.sample_step))
        current_samples_to_check = samples_to_check.copy()

        for detector in self.detectors:
            detections = detector.detect(data, current_samples_to_check)
            evaluation = detector.evaluate(result[file_id].expected_detections,
                                           result[file_id].expected_rebound,
                                           samples_to_check, detections,
                                           tolerance_before=self.tolerance_before,
                                           tolerance_after=self.tolerance_after)
            result[file_id].detections[str(detector)] = detections
            result[file_id].evaluation[str(detector)] = evaluation
            current_samples_to_check = detections

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
