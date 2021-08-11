class Metric:
    pass


class SupervisedMetric(Metric):

    def __init__(self, minimize=True):
        self.direction = "minimize" if minimize else "maximize"

    def as_objective(self, model, X, y=None):
        raise NotImplementedError()

    def __call__(self, y_true, y_pred, *args, **kargs):
        raise NotImplementedError()


class UnsupervisedMetric(Metric):

    def __init__(self, minimize=True):
        self.direction = "minimize" if minimize else "maximize"

    def as_objective(self, model, X, y=None):
        raise NotImplementedError()

    def __call__(self, X, y_pred, *args, **kargs):
        raise NotImplementedError()
