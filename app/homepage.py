"""Streamlit homepage to test pipeline."""
import typing
import streamlit as st

import lps_rvt.types as rvt
import lps_rvt.dataloader as rvt_loader
import lps_rvt.pipeline as rvt_pipeline
import lps_rvt.preprocessing as rvt_preprocessing

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
        with st.expander("Configuração do Teste", expanded=False):
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

    def process(self,
                selected_files: typing.List[int],
                preprocessors: typing.List[rvt_pipeline.PreProcessor]) -> None:
        """
        Processes the selected files using the provided preprocessors and pipeline.
        
        Args:
            selected_files (List[int]): List of selected file IDs to be processed.
            preprocessors (List[PreProcessor]): List of preprocessing functions to be applied.
        """
        pipeline = rvt_pipeline.ProcessingPipeline(preprocessors, [])
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
            selected_files = self.show_dataloader_selection()

            with st.expander("Configuração do Pré-processamento", expanded=True):
                preprocessors = rvt_preprocessing.st_show_preprocessing()

            with st.expander("Configuração do Detector", expanded=True):
                st.write("...")

            if st.button("Executar"):
                show_results = True

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("<h1 style='text-align: center;'>RVT</h1>", unsafe_allow_html=True)
        with col2:
            st.image("./data/logo.png", width=300)

        if show_results:
            self.process(selected_files, preprocessors)

if __name__ == "__main__":
    app = Homepage()
    app.run()
