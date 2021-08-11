from sklearn.manifold import TSNE as SK_TSNE

from models.base.model import UnsupervisedModel
from models.base.parameters import Range, Constant


class TSNE(UnsupervisedModel):
    def __init__(self, n_components=2, n_jobs=None):
        UnsupervisedModel.__init__(self, SK_TSNE, None, parameters={
            "n_components": Range(n_components, 2, minimum=2),
            "n_jobs": Constant(n_jobs, -1),
        })
