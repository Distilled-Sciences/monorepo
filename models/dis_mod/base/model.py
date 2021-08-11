import optuna
import numpy as np

from models.base.metric import Metric


class OPTIMIZE:
    factor = 10
    n_trials = 50
    timeout = 600
    random_state = 42
    verbose = False
    n_warmup_steps = 10


class OptimizableModel:
    def __init__(self, base_model, base_objective, parameters):

        self._base = base_model
        self._objective = base_objective

        self._study = None
        self._paramters = parameters
        self.parameters = {name: param.default for name, param in parameters.items()}
        self._model = self._base(**self.parameters)

    def __select_params__(self, trial, factor):
        params = {}
        for key, param in self._paramters.items():
            params[key] = param.chosen if param.chosen is not None else param.select(key, trial, factor)
        return params

    def __prepare_objective__(self, objective_func, factor, X, y=None):
        def generated_objective(trial):
            model = self._base(**self.__select_params__(trial, factor))
            return objective_func(model, X, y)

        return generated_objective

    def optimize(self,
                 X,
                 y=None,
                 factor=OPTIMIZE.factor,
                 n_trials=OPTIMIZE.n_trials,
                 timeout=OPTIMIZE.timeout,
                 random_state=OPTIMIZE.random_state,
                 verbose=OPTIMIZE.verbose,
                 n_warmup_steps=OPTIMIZE.n_warmup_steps,
                 objective: callable = None,
                 direction=None):

        np.random.seed(random_state)
        optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.ERROR)

        objective = objective if objective is not None else self._objective
        if objective is None:
            raise ValueError(f"{type(self)} cannot be optimized")

        if issubclass(type(objective), Metric):
            opt_direction = objective.direction
            objective = objective.as_objective
        elif direction is not None:
            opt_direction = direction
        else:
            raise ValueError(direction)

        study = optuna.create_study(
            direction=opt_direction,
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps),
        )

        objective_func = self.__prepare_objective__(objective, factor=factor, X=X, y=y)
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout)

        self.parameters = study.best_params
        self._model = self._base(**self.parameters)
        self._study = study

        return self

    @property
    def objective(self):
        from optuna.visualization import plot_optimization_history
        from IPython.display import display
        if self._study is None:
            raise ValueError("This model has not been optized")
        display(plot_optimization_history(self._study))

    @property
    def parameter_importance(self):
        from optuna.visualization import plot_param_importances
        from IPython.display import display
        if self._study is None:
            raise ValueError("This model has not been optized")
        display(plot_param_importances(self._study))

    @property
    def parameter_slices(self):
        from optuna.visualization import plot_slice
        from IPython.display import display
        if self._study is None:
            raise ValueError("This model has not been optized")
        display(plot_slice(self._study))


class SupervisedModel(OptimizableModel):
    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def transform(self, X, y):
        return self._model.transform(X, y)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)


class UnsupervisedModel(OptimizableModel):
    def fit(self, X):
        self._model.fit(X)
        return self

    def transform(self, X):
        return self._model.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)



