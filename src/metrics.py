import enum
import pandas as pd
import numpy as np

class Metric(enum.Enum):
    """Enumeration representing metrics for model analysis.

    This enumeration provides a set of metrics commonly used for evaluating multiclasses models and
        a eval method to perform this evaluation.
    """

    DETECTION_PROBABILITY = 0
    FALSE_ALARM_RATE = 1
    FALSE_DISCOVERY_RATE = 2


    def __str__(self):
        """Return the string representation of the Metric enum."""
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

    def apply(self, cm):

        tp, fp, fn, tn = np.array(cm).ravel()

        match self:

            case Metric.DETECTION_PROBABILITY:
                return tp/(fp + tp)
            
            case Metric.FALSE_ALARM_RATE:
                return fp/(tn + fp)

            case Metric.FALSE_DISCOVERY_RATE:
                return fp/(fp + tp)


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