""" Module providing validation features. """

import os
import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import metric
from src.detector import Detector

class Validate():
    """ Class representing validation data. """

    def __init__(self, files: typing.List[str], detectors: typing.List[Detector],\
        root:str = "Result"):

        detectors = [detector.name for detector in detectors]

        self.data = pd.DataFrame(index=detectors, columns=files)
        self.__root = root

        if not os.path.exists(root):
            os.mkdir(root)

    def accumulate(self, detector: str, file: str, cm: np.ndarray) -> None:
        """ Add confusion matrix to class data.

        Args:
            detector (str): Detector used to make confusion matrix.
            file (str): File used to make confusion matrix.
            cm (np.ndarray): confusion matrix.

        Raises:
            ValueError: Detector not in Validate data.
            ValueError: File not in Validate data.
        """

        if not detector in self.data.index:
            raise ValueError(f"Detector {detector} not in Validate data")

        if not file in self.data.columns:
            raise ValueError(f"File {file} not in Validate data")

        self.data.loc[detector, file] = cm

    def build_table(self, metrics_list: typing.List[metric.Metric]) -> pd.DataFrame:
        """ Build table to be shown or saved.

        Args:
            metrics_list (typing.List[metric.Metric]): List of metrics on table.

        Returns:
            pd.DataFrame: Table.
        """

        table = pd.DataFrame()

        for detector in self.data.index:
            for metric_ in metrics_list:
                values = [metric_.apply(matrix)*100 for matrix in self.data.loc[detector, :]]
                table.loc[detector, str(metric_)] = \
                    f"{np.mean(values):.2f} +- {np.std(values):.2f}%"

        path = os.path.join(self.__root, "table.csv")
        table.to_csv(path)

        return table

    def confusion_matrix(self, detector: str) -> None:
        """Save an acumulated confusion matrix of all data.

        Args:
            detector (str): detector used to make confusion matrices.
        """

        cm_big = [[[], []],
                [[], []]]

        for matrix in self.data.loc[detector, :]:
            for (i, j), val in np.ndenumerate(matrix):
                cm_big[i][j].append(val)

        cm_mean = np.array([[np.mean(cm_big[0][0]), np.mean(cm_big[0][1])],
                            [np.mean(cm_big[1][0]), np.mean(cm_big[1][1])]])

        cm_std = np.array([[np.std(cm_big[0][0]), np.std(cm_big[0][1])],
                                [np.std(cm_big[1][0]), np.std(cm_big[1][1])]])

        fig, ax = plt.subplots()
        ax.matshow(cm_mean, cmap="Oranges")

        for (i, j), mean in np.ndenumerate(cm_mean):
            std = cm_std[i, j]
            text = f"{mean:.2f} +- {std:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black")

        ax.axis("off")

        plt.title(f"{detector}")
        path = os.path.join(self.__root, f"{detector}.png")
        plt.savefig(path)

        plt.close('all')
        sns.reset_defaults()
