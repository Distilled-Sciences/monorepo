from dis_mod.base.wrappers.base_wrapper import BaseWrapper


class PYODWrapper(BaseWrapper):
    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.model.labels_
