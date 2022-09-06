import numpy as np

from esmace.structure import Structure
from esmace.utils import check_is_fitted


class Discretizer:

    def fit_target_sample(self, x: np.ndarray):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def to_structure(self, x: np.ndarray) -> Structure:
        pass

    def n_features(self) -> int:
        pass

    def discretizer_area(self) -> Structure:
        pass


class TabularDiscretizer(Discretizer):

    def __init__(self, num_bins: int = 10) -> None:
        super().__init__()
        self.num_bins = num_bins

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        bin_start = []
        for feat in range(X.shape[1]):
            feat_bins = [X[:, feat].min()]

            for i in range(1, self.num_bins):
                feat_bins.append(np.quantile(X[:, feat], q=(i + 1) / (self.num_bins + 1)))

            feat_bins.append(X[:, feat].max())
            bin_start.append(feat_bins)

        self.bin_start_ = np.array(bin_start)
        self.n_features_ = len(self.bin_start_)
        self.num_bins_feature_ = np.array([self.num_bins] * self.n_features_)

    def to_obs_bins(self, x: np.ndarray) -> np.ndarray:
        bins = np.array([np.digitize(x[i], self.bin_start_[i]) for i in range(len(x))]).astype(int) - 1

        if not np.all(bins >= 0) and not np.all(bins <= self.num_bins):
            raise ValueError(f'Observation {x} is outside the training ranges, could not convert to bins')

        return bins

    def discretizer_area(self) -> Structure:
        bins = np.zeros((self.n_features_, 2), dtype=int)
        bins[:, 1] = self.num_bins_feature_ - 1
        return Structure(bins)

    def to_structure(self, x: np.ndarray) -> Structure:
        check_is_fitted(self)
        obs_bins = self.to_obs_bins(x)
        bins = np.column_stack((obs_bins, obs_bins)).astype(int)
        return Structure(bins)

    def n_features(self) -> int:
        check_is_fitted(self)
        return self.n_features_
