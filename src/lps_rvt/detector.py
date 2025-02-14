"""
Module to provide a set of detectors for audio signal processing within a pipeline.
"""
import enum
import typing
import argparse
import overrides
import pywt

import numpy as np
import streamlit as st
import streamlit_sortables as ss
import torch

import ml.models.base_model as lps_ml
import ml.models.mlp as lps_mlp
import ml.models.cnn as lps_cnn
import lps_rvt.pipeline as rvt_pipeline
import lps_rvt.preprocessing as rvt_preprocessing
import lps_rvt.ml_loader as rvt_ml

class Threshold(rvt_pipeline.Detector):
    """Detects events based on a simple threshold comparison."""

    def __init__(self, window_size: int, threshold: float):
        super().__init__(threshold)
        self.window_size = window_size

    @overrides.overrides
    def calc_confidence(self, input_data: np.ndarray, sample_to_check: int) -> float:
        """
        Estimate sample confidence as a detection

        Args:
            input_data (np.ndarray): The data to search for events.
            sample_to_check (int): The list of sample indices to check.

        Returns:
            float: confidence
        """
        if sample_to_check + self.window_size <= len(input_data):
            return np.mean(input_data[sample_to_check:sample_to_check + self.window_size])
        return 0


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
        super().__init__(threshold)
        self.ref_window = ref_window
        self.analysis_window = analysis_window

    @overrides.overrides
    def calc_confidence(self, input_data: np.ndarray, sample_to_check: int) -> float:
        """
        Estimate sample confidence as a detection

        Args:
            input_data (np.ndarray): The data to search for events.
            sample_to_check (int): The list of sample indices to check.

        Returns:
            float: confidence
        """
        if sample_to_check - self.ref_window >= 0 and \
                    sample_to_check + self.analysis_window <= len(input_data):

            ref_energy = np.mean(input_data[sample_to_check - \
                                            self.ref_window:sample_to_check] ** 2)
            analysis_energy = np.max(input_data[sample_to_check:sample_to_check + \
                                                self.analysis_window] ** 2)

            if ref_energy == 0:
                return 0

            return analysis_energy / ref_energy

        return 0

    @staticmethod
    def st_config() -> "Energy":
        """
        Configures the energy detector processor through Streamlit's interface.

        Returns:
            Energy: A configured energy detector instance.
        """
        ref_window = st.number_input("Janela de referência (Energy)", min_value=1, value=8000)
        analysis_window = st.number_input("Janela para análise (Energy)", min_value=1, value=80)
        threshold = st.number_input("Limiar de proporção (x referência) (Energy)", value=10)
        return Energy(ref_window, analysis_window, threshold)

class ZScore(rvt_pipeline.Detector):
    """Detects events based on Z-score analysis of signal fluctuations."""

    def __init__(self, ref_window: int, analysis_window: int, threshold: float):
        super().__init__(threshold)
        self.ref_window = ref_window
        self.analysis_window = analysis_window

    @overrides.overrides
    def calc_confidence(self, input_data: np.ndarray, sample_to_check: int) -> float:
        """
        Estimate sample confidence as a detection

        Args:
            input_data (np.ndarray): The data to search for events.
            sample_to_check (int): The list of sample indices to check.

        Returns:
            float: confidence
        """
        if sample_to_check - self.ref_window >= 0 and \
                sample_to_check + self.analysis_window <= len(input_data):

            ref_data = input_data[sample_to_check - self.ref_window:sample_to_check]
            mean_ref = np.mean(ref_data)
            std_ref = np.std(ref_data)

            if std_ref > 0:
                analysis_data = input_data[sample_to_check:sample_to_check + self.analysis_window]
                z_scores = (analysis_data - mean_ref) / std_ref

                return np.max(np.abs(z_scores))

        return 0

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

    def __init__(self, window_size: int, threshold: float,
                 wavelet: str, level: int):
        super().__init__(threshold)
        self.window_size = window_size
        self.wavelet = wavelet
        self.level = level

    @overrides.overrides
    def calc_confidence(self, input_data: np.ndarray, sample_to_check: int) -> float:
        """
        Estimate sample confidence as a detection

        Args:
            input_data (np.ndarray): The data to search for events.
            sample_to_check (int): The list of sample indices to check.

        Returns:
            float: confidence
        """
        if sample_to_check + self.window_size <= len(input_data):
            analysis_data = input_data[sample_to_check: sample_to_check + self.window_size]
            coeffs = pywt.wavedec(analysis_data, self.wavelet, level=self.level)

            detail_coeffs = np.concatenate(coeffs[1:])
            return np.sum(detail_coeffs ** 2)

        return 0

    @staticmethod
    def st_config() -> "Wavelet":
        """
        Configures the Wavelet detector processor through Streamlit's interface.

        Returns:
            Wavelet: A configured Wavelet detector instance.
        """

        wavelet_list = ["haar", "db4", "sym2", "coif2"]

        window_size = st.number_input("Janela de análise", min_value=1, value=1000)
        threshold = st.number_input("Limiar de detecção da Wavelet", min_value=0.0, value=0.1)
        wavelet = st.selectbox("Tipo de Wavelet", wavelet_list)
        level = st.slider("Nível de decomposição", min_value=1, max_value=5, step=1, value=2)
        return Wavelet(window_size, threshold, wavelet, level)

class EnergyBand(rvt_pipeline.Detector):
    """Detects events based on an increase in energy compared
    to a reference window in determined band."""

    def __init__(self, ref_window: int, analysis_window: int, threshold: float,
                 min_freq: float, max_freq: float, order: int):
        super().__init__(threshold)
        self.detector = Energy(ref_window, analysis_window, threshold)
        self.preprocessing = rvt_preprocessing.BandPass(min_freq, max_freq, order)

    @overrides.overrides
    def calc_confidence(self, input_data: np.ndarray, sample_to_check: int) -> float:
        """
        Estimate sample confidence as a detection

        Args:
            input_data (np.ndarray): The data to search for events.
            sample_to_check (int): The list of sample indices to check.

        Returns:
            float: confidence
        """
        input_data = self.preprocessing.process(8000, input_data)
            # pylint: disable=W0511 #TODO fix this to work with diferent fs
        return self.detector.calc_confidence(input_data, sample_to_check)

    @staticmethod
    def st_config() -> "EnergyBand":
        """
        Configures the energy detector processor through Streamlit's interface.

        Returns:
            Energy: A configured energy detector instance.
        """
        ref_window = st.number_input("Janela de referência (EnergyBand)", min_value=1, value=8000)
        analysis_window = st.number_input("Janela de análise (EnergyBand)", min_value=1, value=80)
        threshold = st.number_input("Limiar de proporção (x referência) (EnergyBand)", value=10)
        min_freq = st.number_input("Frequência mínima de corte (Hz) (EnergyBand)",
                                                                        min_value=1, value=1)
        max_freq = st.number_input("Frequência máxima de corte (Hz) (EnergyBand)",
                                                                        min_value=1, value=3999)
        order = st.slider("Ordem do filtro (EnergyBand)", min_value=1, max_value=10, value=1)
        return EnergyBand(ref_window, analysis_window, threshold, min_freq, max_freq, order)

class MLModels(enum.Enum):
    MLP = 0
    CNN = 1

    def __str__(self):
        return self.name.split(".")[-1].lower()

class ML(rvt_pipeline.Detector):
    """Detects events using an machine learning models."""

    def __init__(self, model_type: MLModels, threshold: float, model_file: str = None):
        super().__init__(threshold)
        self.model_type = model_type
        self.loaded = False
        self.model_file = model_file or f"./data/ml/models/{model_type}_spectrogram.pkl"
        self.device = None
        self.model = None
        self.n_samples = 2000
        self.transform = None

    def _load(self):
        if self.loaded:
            return
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = lps_ml.BaseModel.load(self.model_file).to(self.device)
        self.model.eval()
        self.loaded = True
        self.transform = rvt_ml.SpectrogramTransform().to(self.device)

    @overrides.overrides
    def calc_confidence(self, input_data: np.ndarray, sample_to_check: int) -> float:
        """
        Estimate sample confidence as a detection

        Args:
            input_data (np.ndarray): The data to search for events.
            sample_to_check (int): The sample index to check.

        Returns:
            float: confidence
        """
        self._load()
        data_tensor = torch.tensor(input_data[sample_to_check:sample_to_check+self.n_samples], dtype=torch.float32, device=self.device).unsqueeze(0)
        data_tensor = self.transform(data_tensor).unsqueeze(0)
        with torch.no_grad():
            confidence = self.model(data_tensor).item()
        return confidence

    @staticmethod
    def st_config() -> 'ML':
        """
        Configures the detector processor through Streamlit's interface.

        Returns:
            ML: A configured ML instance.
        """
        models = MLModels
        model_type = st.selectbox("Modelo", models, format_func=lambda x: x.name, index=0)
        threshold = st.number_input("Limiar", value=0.3)
        return ML(MLModels(model_type), threshold)


def st_show_detect() -> typing.List[rvt_pipeline.Detector]:
    """Displays the detector configuration interface and returns the configured detectors."""
    available_detectors = {
        "Threshold": Threshold,
        "Energy": Energy,
        "Z-score": ZScore,
        "Wavelet": Wavelet,
        "EnergyBand": EnergyBand,
        "ML": ML,
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
                                        default=["Wavelet"])

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
    detector_group.add_argument("--wavelet_threshold", type=float, default=0.3,
                                help="Threshold factor for Wavelet Detector.")
    detector_group.add_argument("--wavelet_type", type=str, default="db4",
                                help="Wavelet Type for Wavelet Detector.")
    detector_group.add_argument("--wavelet_level", type=int, default=1,
                                help="Decomposition Level for Wavelet Detector.")

    # EnergyBand parameters
    detector_group.add_argument("--energyband-ref_window", type=int, default=8000,
                                help="Reference window size for the EnergyBand Detector.")
    detector_group.add_argument("--energyband-analysis-window", type=int, default=80,
                                help="Analysis window size for the EnergyBand Detector.")
    detector_group.add_argument("--energyband-threshold", type=float, default=10.0,
                                help="Threshold factor for the EnergyBand Detector.")
    detector_group.add_argument("--energyband-min-freq", type=float, default=1.0,
                                help="Minimum frequency for the EnergyBand Detector.")
    detector_group.add_argument("--energyband-max-freq", type=float, default=3999.0,
                                help="Maximum frequency for the EnergyBand Detector.")
    detector_group.add_argument("--energyband-order", type=int, default=1.0,
                                help="Order for the EnergyBand Detector.")

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
        "Wavelet": lambda: Wavelet(args.wavelet_window, args.wavelet_threshold,
                                   args.wavelet_type, args.wavelet_level),
        "EnergyBand": lambda: EnergyBand(args.energyband_ref_window,
                                        args.energyband_analysis_window,
                                        args.energyband_threshold,
                                        args.energyband_min_freq,
                                        args.energyband_max_freq,
                                        args.energyband_order)
    }

    for detector in args.detectors or []:
        processor = available_detectors.get(detector, lambda: None)()
        if processor:
            detectors.append(processor)

    return detectors
