from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from dis_mod.base.metric import UnsupervisedMetric
import numpy as np


class Silhouette(UnsupervisedMetric):
    def __init__(self, metric="euclidean", sample_size=None, random_state=42, cached=True):
        super().__init__(minimize=False)
        self.metric = metric
        self.sample_size = sample_size
        self.random_state = random_state
        self._X_distances = None
        self.cached = cached

    def as_objective(self, model, X, y=None):
        if self._X_distances is None:
            if self.cached:
                self._X_distances = cdist(X, X, self.metric)
            else:
                self._X_distances = X

        metric = "precomputed" if self.cached else self.metric

        labels = model.fit_transform(X)
        score = silhouette_score(self._X_distances,
                                 labels,
                                 metric=metric,
                                 sample_size=self.sample_size,
                                 random_state=self.random_state)
        return score

    def __call__(self, X, y_pred, *args, **kwargs):
        score = silhouette_score(X, y_pred, metric=self.metric, sample_size=self.sample_size,
                                 random_state=self.random_state)
        return score
