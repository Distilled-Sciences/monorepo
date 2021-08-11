from sklearn.mixture import BayesianGaussianMixture as SK_BGM

from dis_mod.base.model import UnsupervisedModel
from dis_mod.base.parameters import Categorical, Range, Constant
from dis_mod.base.wrappers.sklearn_wrapper import SKLearnWrapper
from dis_mod.metrics.clustering.variance_ratio import VarianceRatio


class BayesianGaussianMixture(UnsupervisedModel):
    def __init__(self,
                 covariance_type=None,
                 n_components=None,
                 reg_covar=None,
                 tol=None,
                 weight_concentration_prior_type=None,
                 random_state=None):
        base = SKLearnWrapper(SK_BGM)
        UnsupervisedModel.__init__(self, base_model=base, base_objective=VarianceRatio(), parameters={
            "weight_concentration_prior_type": Categorical(weight_concentration_prior_type, "dirichlet_process",
                                                           ['dirichlet_process', 'dirichlet_distribution']),
            "covariance_type": Categorical(covariance_type, "full", ['spherical', 'tied', 'diag', 'full']),
            "n_components": Range(n_components, default=2, minimum=2),
            "reg_covar": Range(reg_covar, default=1e-6),
            "tol": Range(tol, default=1e-3),
            "random_state": Constant(random_state, default=42),
        })
