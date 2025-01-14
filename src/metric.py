import enum
import numpy as np

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

    def apply(self, cm: np.ndarray):

        tp, fn, fp, tn = np.array(cm).ravel()

        if self == Metric.DETECTION_PROBABILITY:
            return tp/(tp + fn)
            
        if self == Metric.FALSE_ALARM_RATE:
            return fp/(tn + fp)

        if self == Metric.FALSE_DISCOVERY_RATE:
            return fp/(fp + tp)
        
        if self == Metric.PRECISION:
            return tp/(tp + fp)

