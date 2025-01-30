""" Module providing validation features. """

import os
import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import rvt.metric as metric
from rvt.detector import Detector

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

    def build_table(self, metrics_list: typing.List[metric.Metric],
                    params: dict, detector_tag: str = 'detector') -> pd.DataFrame:
        """ Build table to be shown or saved.

        Args:
            metrics_list (typing.List[metric.Metric]): List of metrics on table.
            params (dict): dictionary of detector parameters.
            detector_tag (str): detector tag for csv name.

        Returns:
            pd.DataFrame: Table.
        """
        
        # Modificado para colocar os parametros como colunas da tabela, e
        # montar a tabela com os testes de cada conjunto de parametros.
        path = os.path.join(self.__root, f"{detector_tag}_results_by_file.csv")

        if os.path.exists(path):
            table = pd.read_csv(path)
        else:
            table = pd.DataFrame()

        new_row = {"detector": ""}
        new_row.update(params)

        for detector in self.data.index:
            # sum_matrix = np.zeros((2, 2))
            
            # for matrix in self.data.loc[detector, :]:
            #     print(f"Matriz final: {sum_matrix.shape}")
            #     print(f"Matriz a ser somada: {matrix.shape}")
            #     sum_matrix += matrix
            
            for metric_ in metrics_list:
                # print(f"Matriz final: {sum_matrix}")
                # value = metric_.apply(sum_matrix) * 100
                values = [metric_.apply(matrix) * 100 for matrix in self.data.loc[detector, :]]
                new_row[str(metric_)] = f"{np.mean(values):.2f} +- {np.std(values):.2f}%"
                # new_row[str(metric_)] = f"{value:.2f}"
                new_row["detector"] = detector.split(" - ")[0]
                
        for col in new_row.keys():
            if col not in table.columns:
                table[col] = pd.NA

        param_columns = [col for col in new_row.keys() if col not in ["detector"] +
                        [str(m) for m in metrics_list]]

        if not table.empty:
            is_duplicate = (
                (table["detector"] == new_row["detector"]) &
                table[param_columns].eq(pd.Series(new_row)).all(axis=1)
            ).any()
            
            if not is_duplicate:
                table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)
        else:
            table = pd.DataFrame([new_row])
            
        metric_columns = [str(m) for m in metrics_list]
        param_columns = [col for col in table.columns if col not in ["detector"] + metric_columns]
    
        ordered_columns = ["detector"] + param_columns + metric_columns
        table = table[ordered_columns]

        table.to_csv(path, index=False)
        
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
