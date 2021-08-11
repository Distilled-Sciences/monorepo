import umap

from models.base.model import UnsupervisedModel
from models.base.parameters import Range


class UMAP(UnsupervisedModel):
    def __init__(self, n_components=2):
        UnsupervisedModel.__init__(self, umap.UMAP, None, parameters={
            "n_components": Range(n_components, 2, minimum=2),
        })
