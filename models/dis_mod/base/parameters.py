class Categorical:
    def __init__(self, chosen, default=None, choices=None):
        self.chosen = chosen
        self.default = default
        self.choices = choices

    def select(self, name, trial, factor):
        return trial.suggest_categorical(name, self.choices)


class Range:
    def __init__(self, chosen, default=None, minimum=None, maximum=None):
        self.chosen = chosen
        self.default = default
        self.minimum = minimum
        self.maximum = maximum

    def min(self, factor):
        result = self.minimum if self.minimum is not None else self.default / factor
        return int(result) if isinstance(self.default, int) else result

    def max(self, factor):
        result = self.maximum if self.maximum is not None else self.default * factor
        return int(result) if isinstance(self.default, int) else result

    def select(self, name, trial, factor):
        if isinstance(self.default, int):
            return trial.suggest_int(name, self.min(factor), self.max(factor))
        else:
            return trial.suggest_uniform(name, self.min(factor), self.max(factor))


class Constant:
    def __init__(self, chosen, default):
        self.chosen = chosen
        self.default = default

    def select(self, name, trial, factor):
        return self.chosen if self.chosen is not None else self.default
