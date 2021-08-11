from visualization.base.plot import Plot
import pandas as pd

class CorrelationPlot(Plot):
    def __init__(self, method="pearson", min_periods=1, min_value=None, max_value=None):
        if not callable(method) and method not in ['pearson', 'kendall', 'spearman']:
            raise ValueError(method)

        self.method = method
        self.min_periods = min_periods

        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None, x_label=None, y_label=None, x_ticks=None, y_ticks=None):
        try:
            X = pd.DataFrame(X)
        except:
            raise ValueError(X)

        corr = X.corr(self.method, self.min_periods)
        self.min_value = corr.min() if self.min_value is None else self.min_value
        self.max_value = corr.max() if self.max_value is None else self.max_value

    def transform(self, X, y=None, *args, **kwargs):
        pass

    def fit_transform(self, X, y=None, *args, **kwargs):
        pass
