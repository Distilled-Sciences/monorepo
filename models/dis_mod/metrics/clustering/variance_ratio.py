from sklearn.metrics import calinski_harabasz_score
from models.base.metric import UnsupervisedMetric


class VarianceRatio(UnsupervisedMetric):
    def __init__(self):
        super().__init__(minimize=False)

    def as_objective(self, model, X, y=None):
        return self(X, y_pred=model.fit_transform(X))

    def __call__(self, X, y_pred, *args, **kwargs):
        return calinski_harabasz_score(X, y_pred)
