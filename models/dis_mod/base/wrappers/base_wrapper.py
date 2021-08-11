class BaseWrapper:
    def __init__(self, base):
        self.base = base
        self.model = None

    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        raise NotImplementedError()

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except:
            return getattr(self.model, item)

    def __call__(self, **params):
        new = type(self)(self.base)
        new.model = new.base(**params)
        return new
