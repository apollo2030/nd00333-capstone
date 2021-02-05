from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
class Cleaner(BaseEstimator, TransformerMixin):

    needs_refit = True

    def __init__(self, y_only=False):
        super().__init__()
        self.y_only = y_only

    def fit(self, X, y=None):
        mask = pd.isnull(X)
        self._missing_idxs = np.where(mask)
        self._mean = X[~mask].mean()
        return self

    def transform(self, X, y=None, refit=False):
        if refit:
            self.fit(X, y=y)

        check_is_fitted(self, "_mean")
        X[self._missing_idxs] = self._mean
        return X

    def inverse_transform(self, X):
        X[self._missing_idxs] = np.nan
        return X