"""
Module to define a pipeline for processing audio data,
including preprocessing steps and detection tasks.
"""
import abc
import typing
import threading

import numpy as np
import scipy.signal as scipy
import streamlit as st
import plotly.graph_objs as go
import plotly.io as pio

import lps_rvt.dataloader as rvt

class PreProcessor(abc.ABC):
    """Abstract base class for data preprocessing."""

    @abc.abstractmethod
    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        """
        Processes the input data and returns the processed data.
        
        Args:
            fs (int): Sampling frequency of the input data.
            input_data (np.ndarray): The input data to be processed.
        
        Returns:
            Tuple[int, np.ndarray]: The sampling frequency and the processed data.
        """

class Detector(abc.ABC):
    """Abstract base class for detection algorithms."""

    @abc.abstractmethod
    def detect(self, input_data: np.ndarray, samples_to_check: typing.List[int]) \
            -> typing.List[int]:
        """
        Detects events in the input data.
        
        Args:
            input_data (np.ndarray): The data to search for events.
            samples_to_check (List[int]): The list of sample indices to check.
        
        Returns:
            List[int]: A list of sample indices where events were detected.
        """

    def __str__(self) -> str:
        """Returns the name of the detector class."""
        return self.__class__.__name__

    @staticmethod
    def evaluate(expected_detections: typing.List[int],
                 expected_rebounds: typing.List[int],
                 samples_to_check: typing.List[int],
                 detect_samples: typing.List[int],
                 tolerance_before: int,
                 tolerance_after: int) -> np.ndarray:
        """
        Evaluates the performance of the detector by comparing detected events with
        expected detections.

        Args:
            expected_detections (List[int]): The list of expected detection points.
            expected_rebounds (List[int]): The list of expected rebound points.
            samples_to_check (List[int]): The list of samples that were checked.
            detect_samples (List[int]): The list of detected sample indices.
            tolerance_before (int): Tolerance range before detection.
            tolerance_after (int): Tolerance range after detection.
        
        Returns:
            np.ndarray: The confusion matrix.
        """
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
    """Stores the results of processing a single file, including detections and evaluations."""

    def __init__(self, file_id: int):
        self.file_id = file_id
        self.fs = None
        self.expected_detections = []
        self.expected_rebounds = []
        self.processed_signal = None
        self.detections = {}
        self.evaluations = {}

    def final_plot(self, filename=None) -> None:
        """
        Generates a Plotly plot with the processed signal and expected detections.

        Args:
            filename (str, optional):
                If provided, the plot is saved as an image with the given filename.
                Otherwise the plot is shown as streamlit elements.
        """

        if filename is not None:
            data_resampled = self.processed_signal
            new_fs = self.fs
            resampling_factor = 1
        else:
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

        trace_detections = go.Scatter(
            x=[None], y=[None], mode='lines',
            name="Expected Detections",
            line=dict(color="darkgreen", dash="dot", width=1)
        )

        layout = go.Layout(
            title=f"Arquivo {self.file_id}",
            xaxis=dict(title="Time (seconds)"),
            yaxis=dict(title="Amplitude"),
            showlegend=True,
            shapes=shapes
        )

        fig = go.Figure(data=[trace_signal, trace_detections], layout=layout)

        if filename is None:
            st.plotly_chart(fig)
        else:
            pio.write_image(fig, filename)

class ProcessingPipeline:
    """Manages the full data processing pipeline, applying preprocessing and detection to files."""

    def __init__(self,
                 preprocessors: typing.List[PreProcessor],
                 detectors: typing.List[Detector]) -> None:
        self.preprocessors = preprocessors
        self.detectors = detectors
        self.loader = rvt.DataLoader()
        self.margin = 2000
        self.sample_step = 1
        self.tolerance_before=self.sample_step
        self.tolerance_after=self.sample_step

    def process_file(self, file_id: int, result: dict) -> None:
        """
        Processes a single file with the preprocessing steps and detectors.

        Args:
            file_id (int): The ID of the file to process.
            result (dict): The dictionary to store the results for each file.
        """
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
        """
        Applies the processing pipeline to a list of files in parallel.

        Args:
            files (List[int]): The list of file IDs to process.

        Returns:
            dict: The results for all processed files.
        """
        result = {}
        threads = []

        for file_id in files:
            thread = threading.Thread(target=self.process_file, args=(file_id, result))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return result
