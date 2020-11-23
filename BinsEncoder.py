import numpy as np
from scipy import stats


class BinsEncoder:
    _strategies = {
        'min': lambda ind, x: np.min(x),
        'max': lambda ind, x: np.max(x),
        'mean': lambda ind, x: np.mean(x),
        'median': lambda ind, x: np.median(x),
        'mode': lambda ind, x: stats.mode(x)[0][0],
        'index': lambda ind, x: ind + 1
    }

    def __init__(self, strategy):
        """
        Encoder for bins (e.g. for discretized real feature).

        :param strategy:
            Method used to encode each value in bin.

            min
                Encode each bin with its minimum value.
            max
                Encode each bin with its maximum value.
            mean
                Encode each bin with its mean value.
            median
                Encode each bin with its median value.
            mode
                Encode each bin with its mode value.
            index
                Encode each bin with its index.
        """
        assert strategy in self._strategies, f'Strategy should be one of {list(self._strategies.keys())}'
        self.strategy = strategy
        self.encode_strategy = self._strategies[self.strategy]

    def _convert_X_to_bins(self, X, labels):
        n_bins = np.max(labels) + 1

        bins = [[] for _ in range(n_bins)]
        bins_indices = []

        for x, label in zip(X, labels):
            bins_indices.append(len(bins[label]))  # Save index of current element in bins array
            bins[label].append(x)

        return bins, bins_indices

    def _convert_bins_to_X(self, bins, bins_indices, labels):
        X = []

        for bin_index, label in zip(bins_indices, labels):
            X.append(bins[label][bin_index])

        return np.array(X, dtype=float)

    def fit(self, X, labels=None, bins_borders=None):
        """
        Fit encoder.

        :param X:
            It should be either array-like or list of array-likes.
            In the first case `labels` parameter should be provided. In this
            case `X` represents a feature array and `labels` represents bins
            labels.
            In the second case `X` represents splitted bins.
        :param labels:
            Labels for bins.
        :param bins_borders:
            Borders for bins. Needed for correct processing of empty bins.
            If bin is empty than encode strategy is applied to array of two
            elements [low_border, high_border].
        :return:
            None
        """
        if len(X) == 0:  # TODO: Do i need this?
            return

        if labels is None:
            bins = X
        else:
            bins, _ = self._convert_X_to_bins(X, labels)

        self.bins_encoding_ = np.empty(len(bins), dtype=float)
        if bins_borders is None:
            iter_object = bins
        else:
            iter_object = zip(bins, bins_borders[:-1], bins_borders[1:])

        for i, cur_bin in enumerate(iter_object):
            if bins_borders is not None:
                cur_bin, low_border, high_border = cur_bin

            assert np.ndim(cur_bin) == 1, 'Number of dimensions of each bin should be 1'

            if len(cur_bin) == 0:
                assert bins_borders is not None, 'impossible to encode bin: bins_borders is None and some bin is empty'
                self.bins_encoding_[i] = self.encode_strategy(i, [low_border, high_border])
            else:
                self.bins_encoding_[i] = self.encode_strategy(i, cur_bin)

    def transform(self, X, labels=None):
        """
        Transform X with encoder.

        :param X:
            It should be either array-like or list of array-likes.
            In the first case `labels` parameter should be provided. In this
            case `X` represents a feature array and `labels` represents bins
            labels.
            In the second case `X` represents splitted bins.
        :param labels:
            Labels for bins.
        :return:
            Encoded X.
        """
        if labels is None:
            bins = X
        else:
            bins, bins_indices = self._convert_X_to_bins(X, labels)

        assert len(bins) <= len(self.bins_encoding_), 'Number of bins in fitted X is less than in given X'

        bins_enc = []
        for i, cur_bin in enumerate(bins):
            bins_enc += [
                np.full_like(cur_bin, self.bins_encoding_[i], dtype=self.bins_encoding_.dtype)
            ]

        if labels is None:
            return bins_enc

        X_enc = self._convert_bins_to_X(bins_enc, bins_indices, labels)
        return X_enc

    def fit_transform(self, X, labels=None, bins_borders=None):
        """
        Fit encoder and transform X with it.

        :param X:
            It should be either array-like or list of array-likes.
            In the first case `labels` parameter should be provided. In this
            case `X` represents a feature array and `labels` represents bins
            labels.
            In the second case `X` represents splitted bins.
        :param labels:
            Labels for bins.
        :param bins_borders:
            Borders for bins. Needed for correct processing of empty bins.
            If bin is empty than encode strategy is applied to array of two
            elements [low_border, high_border].
        :return:
            Encoded X.
        """
        self.fit(X, labels, bins_borders)
        return self.transform(X, labels)
