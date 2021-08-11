
from sklearn.decomposition import KernelPCA as SK_KernelPCA

from dis_mod.base.model import UnsupervisedModel
from dis_mod.base.parameters import Range, Categorical, Constant


class KernelPCA(UnsupervisedModel):
    def __init__(self, n_components=2, kernel=None, n_jobs=None):
        UnsupervisedModel.__init__(self, SK_KernelPCA, None, parameters={
            "n_components": Range(n_components, 2, minimum=2),
            "kernel": Categorical(kernel, "linear", ["linear", "poly", "rbf", "sigmoid", "cosine"]),
            "n_jobs": Constant(n_jobs, -1)
        })
