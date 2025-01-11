import os
import enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

class Metric(enum.Enum):
    """Enumeration representing metrics for model analysis.

    This enumeration provides a set of metrics commonly used for evaluating multiclasses models and
        a eval method to perform this evaluation.
    """

    DETECTION_PROBABILITY = 0
    FALSE_ALARM_RATE = 1
    FALSE_DISCOVERY_RATE = 2
    PRECISION = 3

    def __str__(self):
        """Return the string representation of the Metric enum."""
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

    def apply(self, cm):

        tp, fn, fp, tn = np.array(cm).ravel()

        if self == Metric.DETECTION_PROBABILITY:
            return tp/(fp + tp)
            
        if self == Metric.FALSE_ALARM_RATE:
            return fp/(tn + fp)

        if self == Metric.FALSE_DISCOVERY_RATE:
            return fp/(fp + tp)
        
        if self == Metric.PRECISION:
            return tp/(tp + fn)

class Validate():

    def __init__(self):
        self.dict = {}

    def accumulate(self, identifier, cm):

        if not (identifier in self.dict):
            self.dict[identifier] = []

        self.dict[identifier].append(cm)

    def build_table(self, metrics_list):
        
        table = {
            "detectors": []
        }

        for metric in metrics_list:
            table[str(metric)] = []

        for identifier, cm_list in self.dict.items():
            table['detectors'].append(identifier)

            for metric in metrics_list:
                values = [metric.apply(cm)*100 for cm in cm_list]
                table[str(metric)].append(f"{np.mean(values):.2f} +- {np.std(values):.2f}%")
    

        df = pd.DataFrame(table)

        return df

    def confusion_matrix(self, identifier: str, root: str):
        
        tp_big = []
        fn_big = []
        fp_big = []
        tn_big = []

        for cm in self.dict[identifier]:
            tp, fn, fp, tn = np.array(cm).ravel()
            tp_big.append(tp)
            fn_big.append(fn)
            fp_big.append(fp)
            tn_big.append(tn)
        
        cm_big = np.array([[np.mean(tp_big), np.mean(fn_big)],
                [np.mean(fp_big), np.mean(tn_big)]])
        
        cm_uncertain = np.array([[np.std(tp_big), np.std(fn_big)],
                        [np.std(fp_big), np.std(tn_big)]])

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm_big, cmap="Oranges", alpha=0.8) # TODO check here

        for (i, j), val in np.ndenumerate(cm_big):
            cm_uncertain_value = cm_uncertain[i, j]
            text = f"{val:.2f} +- {cm_uncertain_value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="black")

        ax.axis("off")

        plt.title(f"{identifier}")
        path = os.path.join(root,f"{identifier}.png")
        plt.savefig(path)

        plt.close('all')
        sns.reset_defaults()