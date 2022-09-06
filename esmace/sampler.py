from typing import Callable, Tuple

import numpy as np

from esmace.discretizer import Discretizer
from esmace.structure import Structure
from esmace.utils import check_is_fitted


class Sampler:

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.predict_fn = predict_fn

    def sample(self, structure: Structure, n_points: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplemented()

    def initial_sampling(self, structure: Structure, n_points: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.sample(structure, n_points)

    def dataset_fit(self, X: np.ndarray) -> None:
        pass

    def fit_discretizer(self, discretizer: Discretizer) -> None:
        pass


class TabularSampler(Sampler):

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], seed=42) -> None:
        super().__init__(predict_fn)
        self.random_state = np.random.RandomState(seed)

    def sample(self, structure: Structure, n_points: float) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)
        X = self.random_state.uniform(size=(n_points, self.n_features_))
        X_scaled = self._project(structure, X)
        y = self.predict_fn(X_scaled)
        return X_scaled, y

    def _project(self, structure: Structure, X: np.ndarray):
        bins = structure.bins
        bin_start = self.discretizer_.bin_start_
        min_val = bin_start[self.feat_arange_, bins[:, 0]]
        max_val = bin_start[self.feat_arange_, bins[:, 1] + 1]
        range_val = max_val - min_val

        return min_val + (range_val * X)

    def fit_discretizer(self, discretizer: Discretizer):
        check_is_fitted(discretizer)
        self.discretizer_ = discretizer
        self.n_features_ = discretizer.n_features()
        self.feat_arange_ = np.arange(self.n_features_).astype(int)


class CachingTabularSampler(TabularSampler):

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], n_points_cache=10_000, seed=42) -> None:
        super().__init__(predict_fn, seed=seed)
        self.n_points_cache = n_points_cache
        self.previous_area = None

    def _cache_points(self):
        new_area = self.discretizer_.discretizer_area()
        if new_area != self.previous_area:
            self.X_cache_, self.y_cache_ = self.sample(new_area, self.n_points_cache)

    def fit_discretizer(self, discretizer: Discretizer):
        super().fit_discretizer(discretizer)
        self._cache_points()

    def initial_sampling(self, structure: Structure, n_points: float) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self)

        cache_mask = self._filter_points_by_structure(self.X_cache_, structure)
        X_inside = self.X_cache_[cache_mask]
        y_inside = self.y_cache_[cache_mask]

        if n_points is not None:
            X_inside = X_inside[:n_points]
            y_inside = y_inside[:n_points]
        
        return X_inside, y_inside
        
    def _filter_points_by_structure(self, X: np.ndarray, structure: Structure):
        mask = np.ones(len(X)).astype(int)

        for feat in range(self.discretizer_.n_features()):
            low, high = structure.bins[feat]
            mask &= (X[:, feat] >= low) & (X[:, feat] <= high)

        return mask