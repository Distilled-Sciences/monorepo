import numpy as np


class BinEncoder:
    def __init__(self, n_components=None, bins=None, method="linear"):
        if n_components is None and bins is None:
            raise ValueError(f"n_components={n_components}, bins={bins}")

        self.means_ = None

        if bins is None:
            self.n_components = n_components
            self.bins_ = None
        else:
            self.n_components = len(bins)
            self.bins_ = bins

        if method not in ["linear", "even"]:
            raise ValueError(method)

        self.method = method

    def fit(self, X):
        X = np.array(X)
        if not self.bins_:
            if self.method == "linear":
                self.bins_ = np.linspace(X.min(), X.max(), self.n_components)
            elif self.method == "even":
                self.bins_ = np.array([chunk.max() for chunk in np.array_split(sorted(X), self.n_components)])
            else:
                raise ValueError(self.method)

        transformed = self.transform(X)
        self.means_ = [X[transformed == i].mean() for i in range(len(self.bins_))]
        self.variances_ = [X[transformed == i].var() for i in range(len(self.bins_))]

        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, X):
        return np.digitize(X, self.bins_)
