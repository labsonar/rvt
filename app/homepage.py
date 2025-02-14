"""Streamlit homepage to test pipeline."""
import typing
import ast
import numpy as np
import pandas as pd
import streamlit as st

import lps_rvt.rvt_types as rvt
import lps_rvt.dataloader as rvt_loader
import lps_rvt.pipeline as rvt_pipeline
import lps_rvt.preprocessing as rvt_preprocessing
import lps_rvt.detector as rvt_detector
import lps_rvt.metrics as rvt_metrics

class Homepage:
    """Streamlit Homepage"""

    def __init__(self) -> None:
        st.set_page_config(page_title="RVT", layout="wide")
        if "table2_data" not in st.session_state:
            st.session_state["table2_data"] = []

        if "current_result" not in st.session_state:
            st.session_state["current_result"] = None

        if "current_metrics" not in st.session_state:
            st.session_state["current_metrics"] = None

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
        selected_buoys = st.multiselect("Selecione os IDs das Boias",
                                            options=range(1, 6),
                                            default=[])
        selected_subsets = st.multiselect("Selecione os Subconjuntos",
                                            subset_options,
                                            default=[rvt.Subset.TRAIN.name])

        ammunition_types = [rvt.Ammunition[t] for t in selected_ammunition] \
                            if selected_ammunition else None
        buoys = selected_buoys if selected_buoys else None
        subsets = [rvt.Subset[s] for s in selected_subsets] \
                            if selected_subsets else None

        return ammunition_types, buoys, subsets

    def show_pipeline_config(self) -> typing.Tuple[int, int, int, int]:
        """
        Displays the Streamlit sidebar for selecting Pipeline configuration.
        It returns the tuple of parameters files based on the user's choices.
        """
        sample_step = st.number_input("Passo de análise (amostras)", min_value=1, value=80)
        tolerance_before = st.number_input("Tolerância antes (amostras)", min_value=5, value=160)
        tolerance_after = st.number_input("Tolerância depois (amostras)", min_value=5, value=240)
        debounce_steps = st.number_input("Debounce_steps (passos)", min_value=1, value=20)
        return sample_step, tolerance_before, tolerance_after, debounce_steps

    def run(self) -> None:
        """
        Runs the Streamlit app, providing the interface for the user to select files,
        configure preprocessing, and start the processing of files and displaying results.
        """
        show_results = False
        save = False
        show_details = False
        with st.sidebar:

            with st.expander("Arquivos analisados", expanded=False):
                ammunition_types, buoys, subsets = self.show_dataloader_selection()

            with st.expander("Configuração do Pipeline", expanded=False):
                sample_step, tolerance_before, tolerance_after, debounce_steps = \
                        self.show_pipeline_config()

            with st.expander("Configuração do Pré-processamento", expanded=False):
                preprocessors = rvt_preprocessing.st_show_preprocessing()

            with st.expander("Configuração do Detector", expanded=False):
                detectors = rvt_detector.st_show_detect()

            with st.expander("Exibição dos resultados", expanded=False):
                loader_type, plot_type, metrics = rvt_metrics.st_show_metrics_config()

            if plot_type == "Figuras de mérito":
                st.markdown("---")

                col1, col2 = st.columns([1, 1])
                with col1:
                    save = st.toggle("Salvar", value=False)
                with col2:
                    show_details = st.toggle("Exibir detalhes", value=False)

            if st.button("Executar"):
                show_results = True

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("<h1 style='text-align: center;'>RVT</h1>", unsafe_allow_html=True)
        with col2:
            st.image("./data/logo.png", width=300)

        loader = rvt_loader.ArtifactLoader() if loader_type == "Artefato" \
                                            else rvt_loader.DataLoader()
        selected_files = loader.get_files(ammunition_types, buoys, subsets)

        if show_results:
            pipeline = rvt_pipeline.Pipeline(preprocessors=preprocessors,
                                            detectors=detectors,
                                            sample_step=sample_step,
                                            tolerance_before=tolerance_before,
                                            tolerance_after=tolerance_after,
                                            debounce_steps=debounce_steps,
                                            loader=loader)
            st.session_state["current_result"] = pipeline.apply(selected_files)

        if detectors is not None and st.session_state["current_result"] is not None:

            if plot_type == "Plot no tempo":
                for _, result in st.session_state["current_result"].items():
                    result.final_plot(metrics)

            elif show_results:
                table_data = []

                if (st.session_state["current_metrics"] != metrics):
                    st.session_state["current_metrics"] = metrics
                    st.session_state["table2_data"] = []

                exp_metric_dict = {
                    "Experimento": len(st.session_state["table2_data"])
                }
                for metric in metrics:
                    exp_metric_dict[str(metric)] = {
                        'num': [],
                        'den': []
                    }

                for file_id, result in st.session_state["current_result"].items():
                    cm = result.get_cm().ravel()

                    metric_dict = {
                        "Arquivo": file_id
                    }
                    for metric in metrics:
                        r = metric.apply(cm)
                        metric_dict[str(metric)] = f"{r[1]}/{r[2]}"
                        exp_metric_dict[str(metric)]['num'].append(r[1])
                        exp_metric_dict[str(metric)]['den'].append(r[2])

                    table_data.append(metric_dict)

                for metric in metrics:
                    exp_metric_dict[str(metric)] = f"{np.sum(exp_metric_dict[str(metric)]['num'])}/{np.sum(exp_metric_dict[str(metric)]['den'])}"

                df = pd.DataFrame(table_data)

                exp_config = {
                    "Detectors": str([d.get_details() for d in detectors]),
                    "Preprocessors": str([p.get_details() for p in preprocessors]),
                    'Ammunition': str([str(ammu) for ammu in ammunition_types]) if ammunition_types is not None else "",
                    'Buoy': str([str(buoy) for buoy in buoys]) if buoys is not None else "",
                    'Subset': str([str(subset) for subset in subsets]) if subsets is not None else "",
                    "Sample Step": sample_step,
                    "Tolerance Before": tolerance_before,
                    "Tolerance After": tolerance_after,
                    "Debounce Steps": debounce_steps,
                    "Loader": loader_type,
                }

                new_entry = exp_metric_dict | exp_config

                if save:
                    if new_entry not in st.session_state["table2_data"]:
                        st.session_state["table2_data"].insert(0, new_entry)

                table2_data = st.session_state["table2_data"].copy()

                if not save:
                    table2_data.insert(0, new_entry)

                df2 = pd.DataFrame(table2_data)

                n_cols = len(metrics) + 1

                if not show_details:
                    df2 = pd.concat([df2.iloc[:, :n_cols],
                                    df2.iloc[:, n_cols:].loc[:, df2.iloc[:, n_cols:].nunique() > 1]],
                                    axis=1)

                def highlight_rows(s):
                    return ['background-color: #f2f2f2' if i % 2 == 0 else \
                            'background-color: #ffffff' for i in range(len(s))]

                styled_df = df.style \
                    .apply(highlight_rows, axis=0) \
                    .set_properties(**{'border': '1px solid black', 'text-align': 'center',
                                    'vertical-align': 'middle'}) \
                    .set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#4CAF50'),
                                                        ('color', 'white'),
                                                        ('font-weight', 'bold'),
                                                        ('text-align', 'center')]},
                        {'selector': 'tbody td', 'props': [('border', '1px solid black'),
                                                        ('text-align', 'center')]}
                    ])

                styled2_df = df2.style \
                    .apply(highlight_rows, axis=0) \
                    .set_properties(**{'border': '1px solid black', 'text-align': 'center',
                                    'vertical-align': 'middle'}) \
                    .set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#4CAF50'),
                                                        ('color', 'white'),
                                                        ('font-weight', 'bold'),
                                                        ('text-align', 'center')]},
                        {'selector': 'tbody td', 'props': [('border', '1px solid black'),
                                                        ('text-align', 'center')]}
                    ])
                st.dataframe(styled2_df, use_container_width=False, hide_index=True)
                st.markdown("---")
                st.dataframe(styled_df, use_container_width=False, hide_index=True)

            else:

                table2_data = st.session_state["table2_data"].copy()

                if len(table2_data) > 0:

                    df2 = pd.DataFrame(table2_data)

                    n_cols = len(st.session_state["current_metrics"]) + 1

                    if not show_details:
                        df2 = pd.concat([df2.iloc[:, :n_cols],
                                df2.iloc[:, n_cols:].loc[:, df2.iloc[:, n_cols:].nunique() > 1]],
                                axis=1)

                    def highlight_rows(s):
                        return ['background-color: #f2f2f2' if i % 2 == 0 else \
                                'background-color: #ffffff' for i in range(len(s))]

                    styled2_df = df2.style \
                        .apply(highlight_rows, axis=0) \
                        .set_properties(**{'border': '1px solid black', 'text-align': 'center',
                                        'vertical-align': 'middle'}) \
                        .set_table_styles([
                            {'selector': 'thead th', 'props': [('background-color', '#4CAF50'),
                                                            ('color', 'white'),
                                                            ('font-weight', 'bold'),
                                                            ('text-align', 'center')]},
                            {'selector': 'tbody td', 'props': [('border', '1px solid black'),
                                                            ('text-align', 'center')]}
                        ])
                    st.dataframe(styled2_df, use_container_width=False, hide_index=True)

if __name__ == "__main__":
    app = Homepage()
    app.run()
