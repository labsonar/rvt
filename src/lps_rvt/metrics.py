"""
Module to provide a set of metrics for audio signal processing within a pipeline.
"""

import typing
import enum
import numpy as np
import streamlit as st
import streamlit_sortables as ss

class Metric(enum.Enum):
    """Enumeration representing metrics for model analysis.

    This enumeration provides a set of metrics commonly used for evaluating multiclasses models and
        a eval method to perform this evaluation.
    """

    DETECTION_PROBABILITY = 0
    FALSE_ALARM_RATE = 1
    FALSE_DISCOVERY_RATE = 2
    PRECISION = 3
    F1_SCORE = 4
    F2_SCORE = 5
    MISS_RATE = 6
    FALL_OUT = 7
    SELECTIVITY = 8
    FALSE_OMISSION_RATE = 9
    NEGATIVE_PREDICTIVE_VALUE = 10

    def __str__(self):
        """Return the string representation of the Metric enum."""
        return str(self.name).replace("_", " ").title()

    def apply(self, cm: np.ndarray) -> typing.Tuple[float, int, int]:
        """Apply Metric to confusion matrix.

        Args:
            cm (np.ndarray): Confusion matrix in shape 2x2

        Returns:
            float: metric. Recomended to multiply by 100 for better visualization.
        """

        tn, fp, fn, tp = np.array(cm).ravel()

        match self:

            case Metric.DETECTION_PROBABILITY:
                return tp/(tp + fn) if (tp + fn)!=0 else 0, tp, tp+fn

            case Metric.FALSE_ALARM_RATE:
                return fp/(tn + fp) if (tn + fp)!=0 else 0, fp, tn+fp

            case Metric.FALSE_DISCOVERY_RATE:
                return fp/(fp + tp) if (fp + tp)!=0 else 0, fp, fp+tp

            case Metric.PRECISION:
                return tp/(tp + fp) if (tp + fp)!=0 else 0, tp, tp+fp

            case Metric.F1_SCORE:
                return 2*tp / (2*tp + fn + fp) if (2*tp + fn + fp)!=0 else 0, 2*tp, (2*tp + fn + fp)

            case Metric.F2_SCORE:
                return 5*tp / (5*tp + 4*fn + fp) if (5*tp + 4*fn + fp)!=0 else 0, 5*tp, (5*tp + 4*fn + fp)

            case Metric.MISS_RATE:
                return fn / (tp + fn) if (tp + fn)!=0 else 0, fn, (tp + fn)

            case Metric.FALL_OUT:
                return fp / (fp + tn) if (fp + tn)!=0 else 0, fp, (fp + tn)

            case Metric.SELECTIVITY:
                return tn / (fp + tn) if (fp + tn)!=0 else 0, tn, (fp + tn)

            case Metric.FALSE_OMISSION_RATE:
                return fn / (tn + fn) if (tn + fn)!=0 else 0, fn, (tn + fn)

            case Metric.NEGATIVE_PREDICTIVE_VALUE:
                return tn / (tn + fn) if (tn + fn)!=0 else 0, tn, (tn + fn)

def st_show_metrics_config():
    """Displays the metrics configuration interface and returns the set of metrics."""
    loader_type = st.selectbox("Modo de análise",
                    ["Artefato", "Arquivo de teste"],
                    index=0)

    plot_type = st.selectbox("Tipo de exibição",
                    ["Plot no tempo", "Figuras de mérito"],
                    index=0)

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

    available_metrics = {str(metric): metric for metric in list(Metric)}

    selected_metrics = st.multiselect("Selecione as metricas",
                                        list(available_metrics.keys()),
                                        default=[str(Metric.DETECTION_PROBABILITY),
                                                 str(Metric.FALSE_DISCOVERY_RATE)])

    if len(selected_metrics) > 1:
        st.markdown("Defina a ordem")
        ordered_metrics = ss.sort_items(selected_metrics, custom_style=simple_style)
    else:
        ordered_metrics = selected_metrics

    metrics = []
    for metric_name in ordered_metrics:
        metric = available_metrics[metric_name]
        metrics.append(metric)

    return loader_type, plot_type, metrics
