import os
import typing
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import streamlit as st
import streamlit_sortables as ss

import lps_utils.utils as lps_utils

import lps_sp.signal as lps_signal
import lps_rvt.pipeline as rvt_pipeline


class NormalizationProcessor(rvt_pipeline.PreProcessor):
    def __init__(self, norm_type: lps_signal.Normalization):
        self.norm_type = norm_type

    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        return fs, self.norm_type(input_data)

    @staticmethod
    def st_config():
        opts = list(lps_signal.Normalization)
        opts.pop(-1)
        norm_type = st.selectbox("Select Normalization Type", opts,
                        index=lps_signal.Normalization.MIN_MAX_ZERO_CENTERED.value)
        return NormalizationProcessor(norm_type)

class HighPassFilterProcessor(rvt_pipeline.PreProcessor):
    def __init__(self, cutoff_freq: float, order: int):
        self.cutoff_freq = cutoff_freq
        self.order = order

    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        nyquist = 0.5 * fs
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = signal.butter(self.order, normal_cutoff, btype='high', analog=False)
        filtered_data = signal.filtfilt(b, a, input_data)
        return fs, filtered_data

    @staticmethod
    def st_config():
        cutoff_freq = st.number_input("Cutoff Frequency (Hz)", min_value=100, value=1000)
        order = st.slider("Filter Order", min_value=1, max_value=10, value=4)
        return HighPassFilterProcessor(cutoff_freq, order)

class CorrelationProcessor(rvt_pipeline.PreProcessor):
    def __init__(self, reference_file: str):
        self.reference_file = reference_file
        self.reference_fs = None
        self.reference_data = None

    def open(self):
        if self.reference_fs is not None and self.reference_data is not None:
            return

        self.reference_fs, self.reference_data = wav.read(self.reference_file)
        if self.reference_data.ndim > 1:
            self.reference_data = np.mean(self.reference_data, axis=1)

    def process(self, fs: int, input_data: np.ndarray) -> typing.Tuple[int, np.ndarray]:
        self.open()
        if fs != self.reference_fs:
            num_samples = int(len(self.reference_data) * (fs / self.reference_fs))
            reference_data_resampled = signal.resample(self.reference_data, num_samples)
        else:
            reference_data_resampled = self.reference_data

        correlation_result = np.correlate(input_data, reference_data_resampled, mode='valid')
        return fs, correlation_result

    @staticmethod
    def st_config():
        predefined_path = "./data/artifacts"
        files = lps_utils.find_files(predefined_path)

        if not files:
            st.warning("Nenhum arquivo encontrado.")
            return None

        file_map = {os.path.splitext(os.path.basename(f))[0].replace(" ", "--").replace("_", " "): f for f in files}

        selected_name = st.selectbox("Selecione o arquivo de referência:", list(file_map.keys()))

        if not selected_name:
            return None

        return CorrelationProcessor(file_map[selected_name])

def st_show_preprocessing():
    available_processes = {
        "Normalization": NormalizationProcessor,
        "High Pass Filter": HighPassFilterProcessor,
        "Correlation": CorrelationProcessor
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

    selected_processes = st.multiselect("Select Processing Steps", list(available_processes.keys()))

    if len(selected_processes) > 1:
        st.write("Define order")
        ordered_processes = ss.sort_items(selected_processes, custom_style=simple_style)
    else:
        ordered_processes = selected_processes

    preprocessors = []
    for process_name in ordered_processes:
        st.divider()
        st.write(f"Configuration for {process_name}")
        process_class = available_processes[process_name]
        p = process_class.st_config()
        if p is not None:
            preprocessors.append(p)

    return preprocessors
