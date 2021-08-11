from itertools import permutations, product
from collections import defaultdict
import pandas as pd


class NaivePropabilities:
    def __init__(self, columns=None, symmetric=True):
        self.columns = columns
        self.symmetric = symmetric

    def fit(self, df):
        if self.columns is None:
            self.columns = df.columns
        return self

    def transform(self, df):
        freqs = defaultdict(int)
        for column in df.columns:
            for a in df[column].values:
                freqs[(column, a)] += 1

        combos = []
        for (column, value), count in freqs.items():
            combos.append({"column": column, "value": value, "count": count})

        likelihoods = []
        for column, gdf in pd.DataFrame(combos).groupby("column"):
            total = gdf["count"].sum()

            for i in range(len(gdf)):
                current, others = gdf.iloc[i], gdf.iloc[[j for j in range(len(gdf)) if j != i]]
                likelihoods.append((column, current["value"], current["count"] / total))

        return sorted(likelihoods, key=lambda x: x[1], reverse=True)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class ConditionalPropabilities:
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, df):
        if self.columns is None:
            self.columns = df.columns
        return self

    def transform(self, df):
        freqs = defaultdict(int)
        for A, B in set(map(tuple, map(sorted, permutations(df.columns, 2)))):
            if A == B:
                continue

            for a, b in df[[A, B]].values:
                freqs[(A, B, a, b)] += 1

        combos = []
        for (A, B, a, b), count in freqs.items():
            combos.append({"key": (A, B), "quadruple": (A, B, a, b), "count": count})

        likelihoods = set()
        for _, gdf in pd.DataFrame(combos).groupby("key"):
            total = gdf["count"].sum()

            for i in range(len(gdf)):
                current, others = gdf.iloc[i], gdf.iloc[[j for j in range(len(gdf)) if j != i]]
                likelihoods.add((*current["quadruple"], current["count"] / total))

        return likelihoods

    def fit_transform(self, df):
        return self.fit(df).transform(df)