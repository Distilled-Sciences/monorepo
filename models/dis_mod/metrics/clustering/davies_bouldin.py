from sklearn.metrics import davies_bouldin_score
from models.base.metric import UnsupervisedMetric


class DaviesBouldin(UnsupervisedMetric):
    def __init__(self):
        super().__init__(minimize=False)

    def as_objective(self, model, X, y=None):
        return self(X, y_pred=model.fit_transform(X))

    def __call__(self, X, y_pred, *args, **kwargs):
        score = davies_bouldin_score(X, y_pred)
        return score
