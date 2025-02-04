"""Streamlit homepage to test pipeline."""
import typing
import streamlit as st

import lps_rvt.types as rvt
import lps_rvt.dataloader as rvt_loader
import lps_rvt.pipeline as rvt_pipeline
import lps_rvt.preprocessing as rvt_preprocessing
import lps_rvt.detector as rvt_detector

class Homepage:
    """Streamlit Homepage"""

    def __init__(self) -> None:
        st.set_page_config(page_title="RVT", layout="wide")
        self.loader = rvt_loader.DataLoader()

    def show_dataloader_selection(self) -> typing.List[int]:
        """
        Displays the Streamlit sidebar for selecting ammunition types, buoy IDs,
        and subsets. It returns the list of selected files based on the user's choices.

        Returns:
            List[int]: List of selected file IDs based on the filters.
        """
        ammunition_options = [e.name for e in rvt.Ammunition]
        subset_options = [e.name for e in rvt.Subset]

        selected_ammunition = st.multiselect("Selecione os Tipos de Munição",
                                                ammunition_options,
                                                default=[rvt.Ammunition.EXSUP.name])
        selected_buoys = st.multiselect("Selecione os IDs das Boias", options=range(1, 6))
        selected_subsets = st.multiselect("Selecione os Subconjuntos",
                                            subset_options,
                                            default=[rvt.Subset.TRAIN.name])

        ammunition_types = [rvt.Ammunition[t] for t in selected_ammunition] \
                            if selected_ammunition else None
        buoys = selected_buoys if selected_buoys else None
        subsets = [rvt.Subset[s] for s in selected_subsets] \
                            if selected_subsets else None

        return self.loader.get_files(ammunition_types, buoys, subsets)

    def show_pipeline_config(self) -> typing.Tuple[int, int, int, int]:
        """
        Displays the Streamlit sidebar for selecting Pipeline configuration.
        It returns the tuple of parameters files based on the user's choices.
        """
        sample_step = st.number_input("Passo de análise (amostras)", min_value=1, value=20)
        tolerance_before = st.number_input("Tolerância antes (amostras)", min_value=5, value=160)
        tolerance_after = st.number_input("Tolerância depois (amostras)", min_value=5, value=320)
        debounce_steps = st.number_input("Debounce_steps (passos)", min_value=1, value=50)
        return sample_step, tolerance_before, tolerance_after, debounce_steps

    def process(self,
                selected_files: typing.List[int],
                pipeline = rvt_pipeline.Pipeline) -> None:
        """
        Processes the selected files using the provided preprocessors and pipeline.

        Args:
            selected_files (List[int]): List of selected file IDs to be processed.
            pipeline (rvt_pipeline.Pipeline): Pipeline to be applied.
        """

        result_dict = pipeline.apply(selected_files)

        for _, result in result_dict.items():
            result.final_plot()

    def run(self) -> None:
        """
        Runs the Streamlit app, providing the interface for the user to select files,
        configure preprocessing, and start the processing of files and displaying results.
        """
        show_results = False
        with st.sidebar:

            with st.expander("Configuração do Pipeline", expanded=False):
                sample_step, tolerance_before, tolerance_after, debounce_steps = \
                        self.show_pipeline_config()

            with st.expander("Configuração do Teste", expanded=False):
                selected_files = self.show_dataloader_selection()

            with st.expander("Configuração do Pré-processamento", expanded=False):
                preprocessors = rvt_preprocessing.st_show_preprocessing()

            with st.expander("Configuração do Detector", expanded=False):
                detectors = rvt_detector.st_show_detect()

            if st.button("Executar"):
                show_results = True

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("<h1 style='text-align: center;'>RVT</h1>", unsafe_allow_html=True)
        with col2:
            st.image("./data/logo.png", width=300)

        if show_results:
            pipeline = rvt_pipeline.Pipeline(preprocessors=preprocessors,
                                                       detectors=detectors,
                                                       sample_step=sample_step,
                                                       tolerance_before=tolerance_before,
                                                       tolerance_after=tolerance_after,
                                                       debounce_steps=debounce_steps)
            self.process(selected_files=selected_files, pipeline=pipeline)

if __name__ == "__main__":
    app = Homepage()
    app.run()
