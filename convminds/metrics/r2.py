import numpy as np

from brainio.assemblies import DataAssembly, walk_coords
from brainscore_core.metrics import Metric, Score

from convminds.brainscore.metrics.linear_predictivity.metric import linear_regression, ridge_regression
from convminds.brainscore.utils.transformations import CrossValidation


class XarrayR2:
    def __init__(self, stimulus_coord="stimulus_id", neuroid_coord="neuroid_id"):
        self._stimulus_coord = stimulus_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target) -> Score:
        prediction = prediction.sortby([self._stimulus_coord, self._neuroid_coord])
        target = target.sortby([self._stimulus_coord, self._neuroid_coord])

        assert np.array(prediction[self._stimulus_coord].values == target[self._stimulus_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()

        y_true = target.values
        y_pred = prediction.values

        y_mean = np.mean(y_true, axis=0)
        sst = np.sum((y_true - y_mean) ** 2, axis=0)
        sst[sst == 0] = 1e-10
        sse = np.sum((y_true - y_pred) ** 2, axis=0)
        r2 = 1 - (sse / sst)

        neuroid_dims = target[self._neuroid_coord].dims
        return Score(
            r2,
            coords={coord: (dims, values) for coord, dims, values in walk_coords(target) if dims == neuroid_dims},
            dims=neuroid_dims,
        )


class CrossRegressedR2(Metric):
    def __init__(self, regression, crossvalidation_kwargs=None):
        crossvalidation_defaults = dict(train_size=0.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}
        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.r2 = XarrayR2()

    def __call__(self, assembly1: DataAssembly, assembly2: DataAssembly) -> Score:
        return self.cross_validation(assembly1, assembly2, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        return self.r2(prediction, target_test)

    def aggregate(self, scores):
        return scores.median(dim="neuroid")


def linear_r2(*args, regression_kwargs=None, **kwargs):
    regression = linear_regression(regression_kwargs or {})
    return CrossRegressedR2(*args, regression=regression, **kwargs)


def ridge_r2(*args, regression_kwargs=None, **kwargs):
    regression = ridge_regression(regression_kwargs or {})
    return CrossRegressedR2(*args, regression=regression, **kwargs)
