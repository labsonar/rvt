""" 
Module to provide a set of detectors for audio signal processing within a pipeline.
"""
import typing
import argparse
import pywt

import numpy as np
import streamlit as st
import streamlit_sortables as ss

import lps_rvt.pipeline as rvt_pipeline

class Threshold(rvt_pipeline.Detector):
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
    def st_config() -> "Threshold":
        """
        Configures the threshold detector processor through Streamlit's interface.
        
        Returns:
            Threshold: A configured threshold detector instance.
        """
        window_size = st.number_input("Tamanho da janela", min_value=1, value=10)
        threshold = st.number_input("Limiar", value=0.04)
        return Threshold(window_size, threshold)

class Energy(rvt_pipeline.Detector):
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
    def st_config() -> "Energy":
        """
        Configures the energy detector processor through Streamlit's interface.
        
        Returns:
            Energy: A configured energy detector instance.
        """
        ref_window = st.number_input("Janela de referência", min_value=1, value=8000)
        analysis_window = st.number_input("Janela de análise", min_value=1, value=80)
        threshold = st.number_input("Limiar de proporção (x referência)", value=10)
        return Energy(ref_window, analysis_window, threshold)

class ZScore(rvt_pipeline.Detector):
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
    def st_config() -> "ZScore":
        """
        Configures the z-score detector processor through Streamlit's interface.
        
        Returns:
            ZScore: A configured z-score detector instance.
        """
        ref_window = st.number_input("Janela de referência", min_value=1, value=20000)
        analysis_window = st.number_input("Janela de análise", min_value=1, value=80)
        threshold = st.number_input("Limiar do Z-score", min_value=0.1, value=50.0)
        return ZScore(ref_window, analysis_window, threshold)
    
class Wavelet(rvt_pipeline.Detector):
    """Detects events using the Discrete Wavelet Transform."""

    def __init__(self, window_size: int, overlap: float, threshold: float,
                 wavelet: str, level: int):
        self.window_size = window_size
        self.overlap = overlap
        self.threshold = threshold
        self.wavelet = wavelet
        self.level = level
        
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
        # step = int(self.window_size * (1 - self.overlap))
    
        for idx in samples_to_check:
            if idx + self.window_size <= len(input_data):
                analysis_data = input_data[idx: idx + self.window_size]
                coeffs = pywt.wavedec(analysis_data, self.wavelet, level=self.level)
            
                detail_coeffs = np.concatenate(coeffs[1:])  
                energy = np.sum(detail_coeffs ** 2)

                if energy > self.threshold:
                    detected_events.append(idx)

        return detected_events
    
    def sliding_window(self, input_data, window_size, overlap):
        """
        Divides the signal in sliding windows
        """
        step = int(window_size * (1 - overlap))
        for start in range(0, len(input_data) - window_size + 1, step):
            yield input_data[start:start + window_size]

    @staticmethod
    def st_config() -> "Wavelet":
        """
        Configures the Wavelet detector processor through Streamlit's interface.
        
        Returns:
            Wavelet: A configured Wavelet detector instance.
        """

        wavelet_list = ["haar", "db4", "sym2", "coif2"]

        window_size = st.number_input("Janela de análise", min_value=1, value=2000)
        overlap = st.slider("Sobreposição da Janela", min_value=0.0, max_value=1.0, step=0.05)
        threshold = st.number_input("Limiar de detecção da Wavelet", min_value=0.0, value=0.1)
        wavelet = st.selectbox("Tipo de Wavelet", wavelet_list)
        level = st.slider("Nível de decomposição", min_value=1, max_value=5, step=1)
        return Wavelet(window_size, overlap, threshold, wavelet, level)


def st_show_detect() -> typing.List[rvt_pipeline.Detector]:
    """Displays the detector configuration interface and returns the configured detectors."""
    available_detectors = {
        "Threshold": Threshold,
        "Energy": Energy,
        "Z-score": ZScore,
        "Wavelet": Wavelet
    }

    st.markdown(
        """
        <style>
        span[data-baseweb="tag"] {
        background-color: #51A9EA !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    simple_style = """
        .sortable-item {
            background-color: #1EAD2B;
            color: white;
        }
        """

    selected_detectors = st.multiselect("Selecione os detectores",
                                        list(available_detectors.keys()),
                                        default=["Z-score"])

    if len(selected_detectors) > 1:
        st.markdown("Defina a ordem")
        ordered_detectors = ss.sort_items(selected_detectors, custom_style=simple_style)
    else:
        ordered_detectors = selected_detectors

    detectors = []
    for detector_name in ordered_detectors:
        st.divider()
        st.markdown(f"Configuração para {detector_name}")
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

    # Threshold parameters
    detector_group.add_argument("--threshold-window", type=int, default=10,
                                help="Window size for the Threshold.")
    detector_group.add_argument("--threshold-value", type=float, default=0.04,
                                help="Threshold value for the Threshold.")

    # Energy parameters
    detector_group.add_argument("--energy-ref-window", type=int, default=8000,
                                help="Reference window size for the Energy.")
    detector_group.add_argument("--energy-analysis-window", type=int, default=80,
                                help="Analysis window size for the Energy.")
    detector_group.add_argument("--energy-threshold", type=float, default=10.0,
                                help="Threshold factor for the Energy.")

    # ZScore parameters
    detector_group.add_argument("--zscore-ref-window", type=int, default=20000,
                                help="Reference window size for the ZScore.")
    detector_group.add_argument("--zscore-analysis-window", type=int, default=40,
                                help="Analysis window size for the ZScore.")
    detector_group.add_argument("--zscore-threshold", type=float, default=50.0,
                                help="Z-score threshold for the ZScore.")
    
    # Wavelet parameters
    detector_group.add_argument("--wavelet_window", type=int, default=2000,
                                help="Window size for the Wavelet Detector.")
    detector_group.add_argument("--wavelet_overlap", type=float, default=0.2,
                                help="Overlap for Wavelet Detector.")
    detector_group.add_argument("--wavelet_threshold", type=float, default=0.3,
                                help="Threshold factor for Wavelet Detector.")
    detector_group.add_argument("--wavelet_type", type=str, default="db4",
                                help="Wavelet Type for Wavelet Detector.")
    detector_group.add_argument("--wavelet_level", type=int, default=1,
                                help="Decomposition Level for Wavelet Detector.")

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
        "Threshold": lambda: Threshold(args.threshold_window,
                                               args.threshold_value),
        "Energy": lambda: Energy(args.energy_ref_window,
                                         args.energy_analysis_window,
                                         args.energy_threshold),
        "Z-score": lambda: ZScore(args.zscore_ref_window,
                                         args.zscore_analysis_window,
                                         args.zscore_threshold),
        "Wavelet": lambda: Wavelet(args.wavelet_window, args.wavelet_overlap,
                                   args.wavelet_threshold, args.wavelet_type, 
                                   args.wavelet_level)
    }

    for detector in args.detectors or []:
        processor = available_detectors.get(detector, lambda: None)()
        if processor:
            detectors.append(processor)

    return detectors
