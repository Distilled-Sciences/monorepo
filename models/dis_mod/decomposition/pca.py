from sklearn.decomposition import PCA as SK_PCA

from models.base.model import UnsupervisedModel
from models.base.parameters import Range


class PCA(UnsupervisedModel):
    def __init__(self, n_components=2):
        UnsupervisedModel.__init__(self, SK_PCA, None, parameters={
            "n_components": Range(n_components, 2, minimum=2),
        })
