"""MODULE 2 - Data preprocessing & feature engineering (Eqs. 7, 8)."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class Preprocessor:
    def __init__(self, k): self.k = k
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)                                   # Eq. 7  z = (x - mu)/sigma
        Z = self.scaler.transform(X)
        self.sel = SelectKBest(mutual_info_classif, k=min(self.k, X.shape[1])).fit(Z, y)   # Eq. 8 mutual information
        self.mi = self.sel.scores_
        return self
    def scale(self, X): return self.scaler.transform(X)
    def transform(self, X): return self.sel.transform(self.scaler.transform(X))
    def selected(self, names): return [names[i] for i in self.sel.get_support(indices=True)]
