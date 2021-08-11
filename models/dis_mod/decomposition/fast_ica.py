from sklearn.decomposition import FastICA as SK_FastICA

from models.base.model import UnsupervisedModel
from models.base.parameters import Range, Categorical


class FastICA(UnsupervisedModel):
    def __init__(self, n_components=2, tol=None, fun=None):
        UnsupervisedModel.__init__(self, SK_FastICA, None, parameters={
            "n_components": Range(n_components, 2, minimum=2),
            "fun": Categorical(fun, "logcosh", ["logcosh", "exp", "cube"]),
            "tol": Range(tol, 1e-4),
        })
