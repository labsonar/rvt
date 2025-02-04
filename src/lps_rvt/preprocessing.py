""" 
Module to provide a set of preprocessing steps for audio signal processing within a pipeline.
"""
import os
import typing
import argparse

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import streamlit as st
import streamlit_sortables as ss

import lps_utils.utils as lps_utils

import lps_sp.signal as lps_signal
import lps_rvt.pipeline as rvt_pipeline


class Normalization(rvt_pipeline.Preprocessing):
    """Processor for normalizing audio data."""
    def __init__(self, norm_type: lps_signal.Normalization) -> None:
        self.norm_type = norm_type

    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        """
        Processes the input data with the chosen normalization method.
        
        Args:
            fs (int): Sampling frequency of the input data.
            input_data (np.ndarray): The audio signal to be processed.
        
        Returns:
            Tuple[int, np.ndarray]: The sampling frequency and the normalized audio signal.
        """
        return fs, self.norm_type(input_data)

    @staticmethod
    def st_config() -> "Normalization":
        """
        Configures the normalization processor through Streamlit's interface.
        
        Returns:
            Normalization: A configured normalization processor instance.
        """
        opts = list(lps_signal.Normalization)
        opts.pop(-1)
        norm_type = st.selectbox("Selecione o tipo de normalização", opts,
                        index=lps_signal.Normalization.MIN_MAX_ZERO_CENTERED.value)
        return Normalization(norm_type)

class HighPass(rvt_pipeline.Preprocessing):
    """Processor for applying a high-pass filter to the audio signal."""

    def __init__(self, cutoff_freq: float, order: int) -> None:
        self.cutoff_freq = cutoff_freq
        self.order = order

    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        """
        Applies a high-pass filter to the input data.

        Args:
            fs (int): Sampling frequency of the input data.
            input_data (np.ndarray): The audio signal to be processed.

        Returns:
            Tuple[int, np.ndarray]: The sampling frequency and the filtered audio signal.
        """
        nyquist = 0.5 * fs
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = signal.butter(self.order, normal_cutoff, btype='high', analog=False)
        filtered_data = signal.filtfilt(b, a, input_data)
        return fs, filtered_data

    @staticmethod
    def st_config() -> "HighPass":
        """
        Configures the high-pass filter processor through Streamlit's interface.
        
        Returns:
            HighPass: A configured high-pass filter processor instance.
        """
        cutoff_freq = st.number_input("Frequência de corte (Hz)", min_value=100, value=1000)
        order = st.slider("Ordem do filtro", min_value=1, max_value=10, value=4)
        return HighPass(cutoff_freq, order)

class Correlation(rvt_pipeline.Preprocessing):
    """Processor for performing correlation with a reference file."""

    def __init__(self, reference_file: str) -> None:
        self.reference_file = reference_file
        self.reference_fs = None
        self.reference_data = None

    def open(self) -> None:
        """Opens and loads the reference file."""
        if self.reference_fs is not None and self.reference_data is not None:
            return

        self.reference_fs, self.reference_data = wav.read(self.reference_file)
        if self.reference_data.ndim > 1:
            self.reference_data = np.mean(self.reference_data, axis=1)

    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        """
        Processes the input data by performing correlation with the reference data.
        
        Args:
            fs (int): Sampling frequency of the input data.
            input_data (np.ndarray): The audio signal to be processed.
        
        Returns:
            Tuple[int, np.ndarray]: The sampling frequency and the correlation result.
        """
        self.open()
        if fs != self.reference_fs:
            num_samples = int(len(self.reference_data) * (fs / self.reference_fs))
            reference_data_resampled = signal.resample(self.reference_data, num_samples)
        else:
            reference_data_resampled = self.reference_data

        correlation_result = np.correlate(input_data, reference_data_resampled, mode='full')
        return fs, correlation_result

    @staticmethod
    def get_file_map(without_spaces: bool = False) -> typing.Optional[dict]:
        """
        Returns a map of reference files in the predefined path.

        Args:
            without_spaces (bool): Whether to replace spaces with '--' in the filenames.

        Returns:
            dict: A map of filenames to file paths.
        """
        predefined_path = "./data/artifacts"
        files = lps_utils.find_files(predefined_path)

        if not files:
            st.warning("Nenhum arquivo encontrado.")
            return None

        if without_spaces:
            return {os.path.splitext(os.path.basename(f))[0].replace(" ", "--"): f for f in files}

        return {os.path.splitext(os.path.basename(f))[0].replace(" ", "--").replace("_", " "): \
                    f for f in files}

    @staticmethod
    def st_config() -> "Correlation":
        """
        Configures the correlation processor through Streamlit's interface.

        Returns:
            Correlation: A configured correlation processor instance.
        """
        file_map = Correlation.get_file_map(False)

        selected_name = st.selectbox("Selecione o arquivo de referência:", list(file_map.keys()))

        if not selected_name:
            return None

        return Correlation(file_map[selected_name])

def st_show_preprocessing() -> typing.List[rvt_pipeline.Preprocessing]:
    """
    Displays the preprocessing configuration interface and returns the configured processors.

    Returns:
        List[Preprocessing]: A list of configured preprocessing processors.
    """
    available_processes = {
        "Normalization": Normalization,
        "HighPass": HighPass,
        "Correlation": Correlation
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

    selected_processes = st.multiselect("Selecione os passos de processamento",
                                        list(available_processes.keys()))

    if len(selected_processes) > 1:
        st.write("Defina a ordem")
        ordered_processes = ss.sort_items(selected_processes, custom_style=simple_style)
    else:
        ordered_processes = selected_processes

    preprocessors = []
    for process_name in ordered_processes:
        st.divider()
        st.write(f"Configuração para {process_name}")
        process_class = available_processes[process_name]
        p = process_class.st_config()
        if p is not None:
            preprocessors.append(p)

    return preprocessors

def add_preprocessing_options(parser: argparse.ArgumentParser) -> None:
    """
    Adds the preprocessing configuration options to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the options will be added.
    """
    preprocess_group = parser.add_argument_group("Preprocessing Configuration",
                                                 "Define the preprocessors applied to the data.")

    preprocess_group.add_argument("-p", "--preprocessors", nargs="+",
                                   choices=["Normalization", "HighPass", "Correlation"],
                                   help="Select preprocessing steps in the desired order.")

    norm_options = list(lps_signal.Normalization)
    norm_options.pop(-1)
    norm_choices = [str(opt.name) for opt in norm_options]

    norm_group = parser.add_argument_group("Normalization Parameters",
                                           "Configure the Normalization.")

    norm_group.add_argument("--norm-type", type=str, choices=norm_choices,
                            default=lps_signal.Normalization.MIN_MAX_ZERO_CENTERED.name,
                            help="Type of normalization.")

    hp_filter_group = parser.add_argument_group("High-Pass Filter Parameters",
                                                "Configure the HighPass.")

    hp_filter_group.add_argument("--cutoff-freq", type=int, default=1000,
                                 help="Cutoff frequency (Hz).")

    hp_filter_group.add_argument("--order", type=int, choices=range(1, 11), default=4,
                                 help="Filter order.")

    correlation_group = parser.add_argument_group("Correlation Parameters",
                                                  "Configure the Correlation.")

    file_map = Correlation.get_file_map(True)

    correlation_group.add_argument("--reference-name", type=str, choices=list(file_map.keys()),
                                   help="Name of the reference file for correlation.")

def get_preprocessors(args: argparse.Namespace) -> typing.List[rvt_pipeline.Preprocessing]:
    """
    Returns a list of configured preprocessing processors based on the parsed arguments.

    Args:
        args (argparse.Namespace): Parsed arguments containing configuration for preprocessing.

    Returns:
        List[Preprocessing]: A list of configured preprocessing processors.
    """
    file_map = Correlation.get_file_map(True)

    preprocessors = []
    available_processors = {
        "Normalization": lambda: Normalization(lps_signal.Normalization[args.norm_type]),
        "HighPass": lambda: HighPass(args.cutoff_freq, args.order),
        "Correlation": lambda: (
            Correlation(file_map[args.reference_name])
            if args.reference_name and args.reference_name in file_map else None
        ),
    }

    for process in args.preprocessors or []:
        processor = available_processors.get(process, lambda: None)()
        if processor:
            preprocessors.append(processor)

    return preprocessors
