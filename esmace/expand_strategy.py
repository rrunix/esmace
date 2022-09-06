import numpy as np

from esmace.discretizer import Discretizer
from esmace.neighborhood import Neighborhood
from esmace.structure import Structure
from esmace.utils import check_is_fitted


class ExpandStrategy:

    def expand(self, structure: Structure):
        raise NotImplemented()

    def fit_discretizer(self, discritizer: Discretizer, neighborhood: Neighborhood):
        raise NotImplemented()


class StepExpandStrategy(ExpandStrategy):

    def __init__(self, max_step=1) -> None:
        super().__init__()
        self.max_step = max_step

    def _grow_tabular_structure(self, structure, feature, left, right):
        previous_left, previous_right = structure.bins[feature]

        left = left if left is not None else previous_left
        right = right if right is not None else previous_right

        if left > previous_left or right < previous_right:
            raise NotImplementedError("Cannot make candidate smaller")

        bins = np.copy(structure.bins)
        bins[feature] = (left, right)
        return Structure(bins)

    def expand(self, structure: Structure):
        check_is_fitted(self)
        candidates = []

        for feature in range(self.discretizer_.n_features()):
            left, right = structure.bins[feature]

            # Left grow
            for left_new in range(max(0, left - self.max_step), left):
                new_candidate = self._grow_tabular_structure(structure, feature, left_new, right)

                if self.neighborhood_.check_inside(new_candidate):
                    candidates.append(new_candidate)

            # Right grow
            for right_new in range(right + 1,
                                   min(right + self.max_step + 1, self.discretizer_.num_bins_feature_[feature])):
                new_candidate = self._grow_tabular_structure(structure, feature, left, right_new)

                if self.neighborhood_.check_inside(new_candidate):
                    candidates.append(new_candidate)

        return candidates

    def fit_discretizer(self, discritizer: Discretizer, neighborhood: Neighborhood):
        self.discretizer_ = discritizer
        self.neighborhood_ = neighborhood
