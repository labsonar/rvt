""" 
Module to provide a set of detectors for audio signal processing within a pipeline.
"""
import typing
import argparse

import numpy as np
import streamlit as st

import lps_rvt.pipeline as rvt_pipeline


class ThresholdDetector(rvt_pipeline.Detector):
    """Detects events based on a simple threshold comparison."""

    def __init__(self, window_size: int, threshold: float):
        self.window_size = window_size
        self.threshold = threshold

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
        detected_events = []

        for idx in samples_to_check:
            if idx + self.window_size <= len(input_data):
                avg_value = np.mean(input_data[idx:idx + self.window_size])
                if avg_value > self.threshold:
                    detected_events.append(idx)

        return detected_events

    @staticmethod
    def st_config() -> "ThresholdDetector":
        """
        Configures the threshold detector processor through Streamlit's interface.
        
        Returns:
            ThresholdDetector: A configured threshold detector instance.
        """
        window_size = st.number_input("Tamanho da janela", min_value=1, value=10)
        threshold = st.number_input("Limiar", value=0.04)
        return ThresholdDetector(window_size, threshold)

class EnergyDetector(rvt_pipeline.Detector):
    """Detects events based on an increase in energy compared to a reference window."""

    def __init__(self, ref_window: int, analysis_window: int, threshold: float):
        self.ref_window = ref_window
        self.analysis_window = analysis_window
        self.threshold = threshold

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
        detected_events = []

        for idx in samples_to_check:
            if idx - self.ref_window >= 0 and idx + self.analysis_window <= len(input_data):
                ref_energy = np.mean(input_data[idx - self.ref_window:idx] ** 2)
                analysis_energy = np.max(input_data[idx:idx + self.analysis_window] ** 2)

                if ref_energy == 0:
                    continue

                increase_factor = analysis_energy / ref_energy
                if increase_factor > self.threshold:
                    detected_events.append(idx)

        return detected_events

    @staticmethod
    def st_config() -> "EnergyDetector":
        """
        Configures the energy detector processor through Streamlit's interface.
        
        Returns:
            EnergyDetector: A configured energy detector instance.
        """
        ref_window = st.number_input("Janela de referência", min_value=1, value=8000)
        analysis_window = st.number_input("Janela de análise", min_value=1, value=80)
        threshold = st.number_input("Limiar de proporção (x referência)", value=10)
        return EnergyDetector(ref_window, analysis_window, threshold)



class ZScoreDetector(rvt_pipeline.Detector):
    """Detects events based on Z-score analysis of signal fluctuations."""

    def __init__(self, ref_window: int, analysis_window: int, threshold: float):
        self.ref_window = ref_window
        self.analysis_window = analysis_window
        self.threshold = threshold

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
        detected_events = []

        for idx in samples_to_check:
            if idx - self.ref_window >= 0 and idx + self.analysis_window <= len(input_data):
                ref_data = input_data[idx - self.ref_window:idx]
                mean_ref = np.mean(ref_data)
                std_ref = np.std(ref_data)

                if std_ref > 0:
                    analysis_data = input_data[idx:idx + self.analysis_window]
                    z_scores = (analysis_data - mean_ref) / std_ref

                    if np.max(np.abs(z_scores)) > self.threshold:
                        detected_events.append(idx)

        return detected_events

    @staticmethod
    def st_config() -> "ZScoreDetector":
        """
        Configures the z-score detector processor through Streamlit's interface.
        
        Returns:
            ZScoreDetector: A configured z-score detector instance.
        """
        ref_window = st.number_input("Janela de referência", min_value=1, value=20000)
        analysis_window = st.number_input("Janela de análise", min_value=1, value=40)
        threshold = st.number_input("Limiar do Z-score", min_value=0.1, value=50.0)
        return ZScoreDetector(ref_window, analysis_window, threshold)

def st_show_detect() -> typing.List[rvt_pipeline.Detector]:
    """Displays the detector configuration interface and returns the configured detectors."""
    available_detectors = {
        "Threshold": ThresholdDetector,
        "Energy": EnergyDetector,
        "Z-score": ZScoreDetector
    }

    selected_detectors = st.multiselect("Selecione os detectores", list(available_detectors.keys()))

    detectors = []
    for detector_name in selected_detectors:
        st.divider()
        st.write(f"Configuração para {detector_name}")
        detector_class = available_detectors[detector_name]
        d = detector_class.st_config()
        if d is not None:
            detectors.append(d)

    return detectors

def add_detector_options(parser: argparse.ArgumentParser) -> None:
    """
    Adds the detector configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the options will be added.
    """
    detector_group = parser.add_argument_group("Detector Configuration",
                                               "Define the detectors used in the pipeline.")

    detector_group.add_argument("-d", "--detectors", nargs="+",
                                choices=["Threshold", "Energy", "Z-score"],
                                help="Select the detectors to be used.")

    # ThresholdDetector parameters
    detector_group.add_argument("--threshold-window", type=int, default=10,
                                help="Window size for the ThresholdDetector.")
    detector_group.add_argument("--threshold-value", type=float, default=0.04,
                                help="Threshold value for the ThresholdDetector.")

    # EnergyDetector parameters
    detector_group.add_argument("--energy-ref-window", type=int, default=8000,
                                help="Reference window size for the EnergyDetector.")
    detector_group.add_argument("--energy-analysis-window", type=int, default=80,
                                help="Analysis window size for the EnergyDetector.")
    detector_group.add_argument("--energy-threshold", type=float, default=10.0,
                                help="Threshold factor for the EnergyDetector.")

    # ZScoreDetector parameters
    detector_group.add_argument("--zscore-ref-window", type=int, default=20000,
                                help="Reference window size for the ZScoreDetector.")
    detector_group.add_argument("--zscore-analysis-window", type=int, default=40,
                                help="Analysis window size for the ZScoreDetector.")
    detector_group.add_argument("--zscore-threshold", type=float, default=50.0,
                                help="Z-score threshold for the ZScoreDetector.")

def get_detectors(args: argparse.Namespace) -> typing.List[rvt_pipeline.Detector]:
    """
    Returns a list of configured detector instances based on the parsed arguments.

    Args:
        args (argparse.Namespace): Parsed arguments containing detector configurations.

    Returns:
        List[Detector]: A list of configured detector instances.
    """
    detectors = []

    available_detectors = {
        "Threshold": lambda: ThresholdDetector(args.threshold_window,
                                               args.threshold_value),
        "Energy": lambda: EnergyDetector(args.energy_ref_window,
                                         args.energy_analysis_window,
                                         args.energy_threshold),
        "Z-score": lambda: ZScoreDetector(args.zscore_ref_window,
                                         args.zscore_analysis_window,
                                         args.zscore_threshold),
    }

    for detector in args.detectors or []:
        processor = available_detectors.get(detector, lambda: None)()
        if processor:
            detectors.append(processor)

    return detectors
