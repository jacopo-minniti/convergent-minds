from brainscore import metric_registry
from .metric import Accuracy

metric_registry['accuracy'] = Accuracy
