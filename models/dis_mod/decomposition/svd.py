from sklearn.decomposition import TruncatedSVD as SK_TruncatedSVD

from models.base.model import UnsupervisedModel
from models.base.parameters import Range


class SVD(UnsupervisedModel):
    def __init__(self, n_components=2):
        UnsupervisedModel.__init__(self, SK_TruncatedSVD, None, parameters={
            "n_components": Range(n_components, 2, minimum=2),
        })
