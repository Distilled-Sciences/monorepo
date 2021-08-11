from dis_mod.base.wrappers.base_wrapper import BaseWrapper


class SKLearnWrapper(BaseWrapper):
    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        if hasattr(self.model, "transform"):
            return self.model.transform(X)
        else:
            return self.model.predict(X)

    def fit_transform(self, X):
        if hasattr(self.model, "transform"):
            return self.model.fit_transform(X)
        else:
            return self.model.fit_predict(X)
