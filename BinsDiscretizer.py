import numpy as np

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

from BinsEncoder import BinsEncoder


class BinsDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, encode_bins=None, strategy='quantile'):
        """
        Discretizer for real features using binning.

        :param n_bins:
            Number of bins to discretize given feature.
        :param encode_bins:
            Encoding method to apply to each bin. For more details see
            BinsEncoder's `strategy` parameter.
        :param strategy:
            Strategy used to split feature values into bins.

            uniform
                All bins have the same width.
            quantile
                All bins have the same number of observations.
            kmeans
                Cluster observations with kmeans.
        """
        self._strategies = {
            'uniform': lambda X: np.linspace(np.min(X), np.max(X), self.n_bins + 1),
            'quantile': lambda X: np.quantile(X, np.linspace(0, 1, self.n_bins + 1)),
            'kmeans': lambda X: self._get_borders_with_kmeans(X)
        }

        assert n_bins > 1, f'Number of features should be not less than 2, but {n_bins} is given'
        assert strategy in self._strategies, f'Strategy should be one of {list(self._strategies.keys())}'

        self.n_bins = n_bins
        self.encode_bins = encode_bins
        if self.encode_bins is not None:
            self.encoder_ = BinsEncoder(strategy=self.encode_bins)
        self.strategy = strategy
        self.bins_borders_strategy = self._strategies[self.strategy]

    def _get_borders_with_kmeans(self, X):
        uniform_borders = np.linspace(np.min(X), np.max(X), self.n_bins + 1)
        centers_for_init = ((uniform_borders[1:] + uniform_borders[:-1]) / 2).reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.n_bins, init=centers_for_init, n_init=1)

        cluster_centers = kmeans.fit(X.reshape(-1, 1)).cluster_centers_[:, 0]
        cluster_centers.sort()
        bins_borders = np.r_[np.min(X), (cluster_centers[1:] + cluster_centers[:-1]) / 2, np.max(X)]

        return bins_borders

    def _check_X_type(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X

    def _get_labels(self, X):
        X = self._check_X_type(X)
        labels = np.searchsorted(self.bins_borders_, X, side='right')
        labels = np.clip(labels, 1, self.n_bins) - 1
        return labels

    def _get_bins_borders(self, X):
        X = self._check_X_type(X)

        assert np.ndim(X) == 1, 'Number of dimensions of X should be 1'

        bins_borders = self.bins_borders_strategy(X)

        return bins_borders

    def fit(self, X):
        """
        Fit discretizer.

        :param X:
            Feature vector.
        :return:
            None
        """
        self.bins_borders_ = self._get_bins_borders(X)

        if self.encode_bins is not None:
            labels = self._get_labels(X)
            self.encoder_.fit(X, labels, self.bins_borders_)

    def transform(self, X):
        """
        Transform X with discretizer.

        :param X:
            Feature vector.
        :return:
            Binned X.
        """
        labels = self._get_labels(X)

        if self.encode_bins is None:
            return labels

        encoded_bins = self.encoder_.transform(X, labels)
        return encoded_bins

    def fit_transform(self, X):
        """
        Fit discretizer and transform X with it.

        :param X:
            Feature vector.
        :return:
            Binned X.
        """
        self.bins_borders_ = self._get_bins_borders(X)

        labels = self._get_labels(X)
        if self.encode_bins is None:
            return labels

        encoded_bins = self.encoder_.fit_transform(X, labels, self.bins_borders_)
        return encoded_bins

#         self.fit(X)
#         return self.transform(X)
