class Plot:

    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError()
