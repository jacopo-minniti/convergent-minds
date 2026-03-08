from convminds.alignment.metrics.linear_partial_r2 import linear_partial_r2 as partial_r2
from convminds.brainscore.metrics.linear_predictivity.metric import linear_pearsonr, ridge_pearsonr
from convminds.brainscore.metrics.pearson_correlation.metric import PearsonCorrelation
from convminds.brainscore.metrics.cka.metric import CKACrossValidated
from convminds.brainscore.metrics.rdm.metric import RDMCrossValidated
from convminds.brainscore.metrics.accuracy.metric import Accuracy
from .r2 import linear_r2, ridge_r2


def pearsonr(*args, **kwargs):
    return PearsonCorrelation(*args, **kwargs)


def cka(*args, **kwargs):
    return CKACrossValidated(*args, **kwargs)


def rdm(*args, **kwargs):
    return RDMCrossValidated(*args, **kwargs)


def accuracy(*args, **kwargs):
    return Accuracy(*args, **kwargs)


__all__ = [
    "partial_r2",
    "linear_pearsonr",
    "ridge_pearsonr",
    "linear_r2",
    "ridge_r2",
    "pearsonr",
    "cka",
    "rdm",
    "accuracy",
]
