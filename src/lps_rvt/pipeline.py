"""
Module to define a pipeline for processing audio data,
including preprocessing steps and detection tasks.
"""
import os
import abc
import typing
import concurrent.futures as cf

import numpy as np
import pandas as pd
import scipy.signal as scipy
import streamlit as st
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns

import lps_rvt.dataloader as rvt

class Preprocessing(abc.ABC):
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

    def get_details(self) -> typing.Dict:
        """Retorna o dicionario de atributos"""
        return {
            str(self): self.__dict__
        }

    def __str__(self) -> str:
        """Returns the name of the preprocessor class."""
        return self.__class__.__name__

class Detector(abc.ABC):
    """Abstract base class for detection algorithms."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    @abc.abstractmethod
    def calc_confidence(self, input_data: np.ndarray, sample_to_check: int) -> float:
        """
        Estimate sample confidence as a detection

        Args:
            input_data (np.ndarray): The data to search for events.
            sample_to_check (int): The list of sample indices to check.

        Returns:
            float: _description_
        """

    def detect(self, input_data: np.ndarray, samples_to_check: typing.List[int]) \
            -> typing.Tuple[typing.List[int], typing.List[int]]:
        """
        Detects events in the input data.

        Args:
            input_data (np.ndarray): The data to search for events.
            samples_to_check (List[int]): The list of sample indices to check.

        Returns:
            Tuple[
                List[int]: A list of sample indices where events were detected.,
                List[float]: A list of confidence in this detection
                ]
        """
        detected_events = []
        confidences = []

        for idx in samples_to_check:
            confidence = self.calc_confidence(input_data=input_data, sample_to_check=idx)

            if confidence > self.threshold:
                detected_events.append(idx)
                confidences.append(confidence)

        return detected_events, confidences

    def get_details(self) -> typing.Dict:
        """Retorna o dicionario de atributos"""
        return {
            str(self): self.__dict__
        }

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
        self.last_detector = None

    def final_plot(self, metrics, filename=None, resample: bool = True) -> None:
        """
        Generates a Plotly plot with the processed signal and expected detections.

        Args:
            filename (str, optional): If provided, the plot is saved as an image.
                Othewise data is presented as streamlit.
            resample (bool): If True data will be resampled to reduce computacional cost.
        """
        original_samples = len(self.processed_signal)

        if resample and original_samples > 480000:
            num_samples = 480000
            data_resampled = scipy.resample(self.processed_signal, num_samples)

            resampling_factor = original_samples / num_samples
            new_fs = self.fs / resampling_factor
        else:
            data_resampled = self.processed_signal
            new_fs = self.fs
            resampling_factor = 1

        time_axis = [i / new_fs for i in range(len(data_resampled))]

        traces = []

        traces.append(go.Scatter(x=time_axis, y=data_resampled, mode='lines', name='Signal Data',
                                 line=dict(color='darkblue', width=0.5)))

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
                    line=dict(color="darkgreen", width=4, dash="dot")
                )
            )

        traces.append(go.Scatter(
            x=[None], y=[None], mode='lines',
            name="Expected Detections",
            line=dict(color="darkgreen", dash="dot", width=4)
        ))

        for d in self.expected_rebounds:
            new_d = int(d/resampling_factor)
            shapes.append(
                dict(
                    type="line",
                    x0=time_axis[new_d],
                    y0=min(data_resampled),
                    x1=time_axis[new_d],
                    y1=max(data_resampled),
                    line=dict(color="green", width=2, dash="dot")
                )
            )

        traces.append(go.Scatter(
            x=[None], y=[None], mode='lines',
            name="Expected Rebounds",
            line=dict(color="green", dash="dot", width=2)
        ))

        if self.last_detector is not None:
            detections = self.detections[self.last_detector]
            for d in detections:
                new_d = int(d/resampling_factor)
                shapes.append(
                    dict(
                        type="line",
                        x0=time_axis[new_d],
                        y0=min(data_resampled),
                        x1=time_axis[new_d],
                        y1=max(data_resampled),
                        line=dict(color="darkred", width=0.5)
                    )
                )

            traces.append(go.Scatter(
                x=[None], y=[None], mode='lines',
                name="Detections",
                line=dict(color="darkred", width=0.5)
            ))

            title=f"{self.file_id}"
            for metric in metrics:
                _, a, b = metric.apply(self.evaluations[self.last_detector])
                title += f"\t\t\t{str(metric)}: {a}/{b}"

            # title=f"{self.file_id}         Detecção: {tp}/{tp+fn}" \
            #     f"         Falso Positivos: {fp}"
        else:
            title=f"{self.file_id}"

        layout = go.Layout(
            title=title,
            xaxis=dict(title="Time (seconds)"),
            yaxis=dict(title="Amplitude"),
            showlegend=True,
            shapes=shapes
        )

        fig = go.Figure(data=traces, layout=layout)

        if filename is None:
            st.plotly_chart(fig)
        else:
            pio.write_image(fig, filename, width=800, height=600, scale=4)

    def final_cm(self, filename=None) -> None:
        """
        Generates a seaborn plot with the final confusion matrix.

        Args:
            filename (str, optional):
                If provided, the plot is saved as an image with the given filename.
                Otherwise the plot is shown as streamlit elements.
        """
        cm = self.evaluations[self.last_detector]

        df_cm = pd.DataFrame(cm, index=["Falso", "Verdadeiro"], columns=["Falso", "Verdadeiro"])
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax)

        if filename is None:
            st.markdown("<div style='display: flex; justify-content: center;'>",
                        unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            plt.savefig(filename)

    def __str__(self):
        if len(self.evaluations) == 0:
            return super().__str__()
        _, fp, fn,tp= self.evaluations[self.last_detector].ravel()
        return f'{tp}/{fn + tp} -> {fp}'

    def get_cm(self) -> np.ndarray:
        """
        Return the performance of an pipeline.

        Returns:
            np.ndarray: The confusion matrix.
        """
        return self.evaluations[self.last_detector]

class Pipeline:
    """Manages the full data processing pipeline, applying preprocessing and detection to files."""

    def __init__(self,
                 preprocessors: typing.List[Preprocessing],
                 detectors: typing.List[Detector],
                 margin: int = 2000,
                 sample_step: int = 50,
                 tolerance_before: int = None,
                 tolerance_after: int = None,
                 debounce_steps: int = 3,
                 cooldown_samples: int = 16000,
                 loader = rvt.DataLoader()
                 ) -> None:
        self.preprocessors = preprocessors
        self.detectors = detectors
        self.margin = margin
        self.sample_step = sample_step
        self.tolerance_before= tolerance_before if tolerance_before is not None \
                                                else self.sample_step
        self.tolerance_after= tolerance_after if tolerance_after is not None \
                                                else self.sample_step*6
        self.debounce_steps = debounce_steps
        self.cooldown_samples = cooldown_samples
        self.loader = loader

    def _edge_filter(self, samples: typing.List[int],
                     confidence: typing.List[float],
                     cooldown_samples: int = 16000) -> typing.List[int]:
        """
        Filters raw detections into distinct events. Uses minimum burst length
        and a cooldown period to prevent rapid re-triggering.
        """
        if not samples:
            return []

        edges = []
        current_block = [0]
        
        last_event_sample = -cooldown_samples

        for i in range(1, len(samples)):
            break_threshold = self.sample_step * 2
            
            if samples[i] - samples[i - 1] > break_threshold:
                if len(current_block) >= self.debounce_steps:
                    selected_confidence = [confidence[j] for j in current_block]
                    index = np.argmax(selected_confidence)
                    potential_event_sample = samples[current_block[index]]
                    
                    if potential_event_sample - last_event_sample >= cooldown_samples:
                        edges.append(potential_event_sample)
                        last_event_sample = potential_event_sample
                        
                current_block.clear()

            current_block.append(i)

        if len(current_block) >= self.debounce_steps:
            selected_confidence = [confidence[j] for j in current_block]
            index = np.argmax(selected_confidence)
            potential_event_sample = samples[current_block[index]]
            if potential_event_sample - last_event_sample >= cooldown_samples:
                edges.append(potential_event_sample)

        return edges

    def _process_file(self, file_id: int) -> Result:
        """
        Processes a single file with the preprocessing steps and detectors.

        Args:
            file_id (int): The ID of the file to process.

        Returns:
            result (Result): The class Result for defined file.
        """
        result = Result(file_id)


        fs, data = self.loader.get_data(file_id)
        result.expected_detections, result.expected_rebounds = \
                    self.loader.get_critical_points(file_id, fs)

        for preprocessor in self.preprocessors:
            fs, data = preprocessor.process(fs, data)

        result.fs = fs
        result.processed_signal = data

        input_size = len(data)

        samples_to_check = list(range(self.margin, input_size - self.margin, self.sample_step))
        current_samples_to_check = samples_to_check.copy()

        for detector in self.detectors:
            detections, confidence = detector.detect(data, current_samples_to_check)
            detections = self._edge_filter(detections, confidence, cooldown_samples=self.cooldown_samples)
            evaluation = detector.evaluate(result.expected_detections,
                                           result.expected_rebounds,
                                           samples_to_check,
                                           detections,
                                           tolerance_before=self.tolerance_before,
                                           tolerance_after=self.tolerance_after)
            result.evaluations[str(detector)] = evaluation
            result.detections[str(detector)] = detections.copy()
            result.last_detector = str(detector)
            current_samples_to_check = detections

        return file_id, result

    def _export_file(self, file_id: int, output_dir:str, resample: bool = True) -> None:
        _, result = self._process_file(file_id)
        result.final_plot(os.path.join(output_dir, f"{file_id}.png"), resample)

    def apply(self, files: typing.List[int], max_process: int = None) -> dict:
        """
        Applies the processing pipeline to a list of files in parallel.

        Args:
            files (List[int]): The list of file IDs to process.
            max_process (optional[int]): max number of paralell process.

        Returns:
            dict: The results for all processed files.
        """
        result = {}
        max_workers = max_process if max_process is not None else len(files)

        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_file, file_id): file_id for file_id in files}

            for future in cf.as_completed(futures):
                file_id, processed_data = future.result()
                result[file_id] = processed_data

        return result

    def export(self, files: typing.List[int], output_dir: str,
               resample: bool = True, max_process: int = None) -> dict:
        """
        Applies the processing pipeline to a list of files in parallel
            exporting the final plot for each of then.

        Args:
            files (List[int]): The list of file IDs to process.
            output_dir (str): Output directory
            resample (bool): If True data will be resampled to reduce computacional cost
                when printing.
            max_process (optional[int]): max number of paralell process.
        """
        result = {}
        max_workers = max_process if max_process is not None else len(files)

        os.makedirs(output_dir, exist_ok=True)

        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._export_file, file_id, output_dir, resample): file_id \
                                                                            for file_id in files}

            for future in cf.as_completed(futures):
                future.result()

        return result
