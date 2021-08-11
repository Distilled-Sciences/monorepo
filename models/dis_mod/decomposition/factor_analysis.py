from sklearn.decomposition import FactorAnalysis as SK_FactorAnalysis

from models.base.model import UnsupervisedModel
from models.base.parameters import Range


class FactorAnalysis(UnsupervisedModel):
    def __init__(self, n_components=2, tol=None):
        UnsupervisedModel.__init__(self, SK_FactorAnalysis, None, parameters={
            "n_components": Range(n_components, 2, minimum=2),
            "tol": Range(tol, 1e-2),
        })
