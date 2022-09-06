import numpy as np

from esmace.estimation_bounds import hoeffding_bounds
from esmace.grouping_measure import GroupingMeasure
from esmace.structure import Structure
from esmace.utils import dataclass


@dataclass(frozen=True, slots=True)
class Score:
    fixed_component: float

    uncertain_component: float
    uncertain_component_lb: float
    uncertain_component_ub: float

    n_points_estimation: int

    @property
    def score(self):
        return self.fixed_component + self.uncertain_component

    @property
    def lb(self):
        return self.fixed_component + self.uncertain_component_lb

    @property
    def ub(self):
        return self.fixed_component + self.uncertain_component_ub


class Metric:

    def calculate(self, X, y, structure: Structure, previous_estimation: Score = None) -> Score:
        """

        Should be y or the grouped value?
        """
        raise NotImplemented("Method calculate should be implemented")

    def reduce_uncertainty_to(self, estimation: Score, width: float):
        return 100

    def is_estimation(self):
        raise NotImplemented("Method is_estimation should be implemented")


class FidelityMetric(Metric):

    def __init__(self, grouping_measure: GroupingMeasure, p: float = 0.05) -> None:
        super().__init__()
        self.grouping_measure = grouping_measure
        self.p = p

    def calculate(self, X, y, structure: Structure, previous_estimation: Score = None) -> Score:
        new_label = self.grouping_measure.calculate(y)
        hits = new_label == 1

        count = len(new_label)
        avg = np.mean(hits)

        if previous_estimation is None:
            previous_avg = 0
            previous_count = 0
        else:
            previous_avg = previous_estimation.uncertain_component
            previous_count = previous_estimation.n_points_estimation

        new_count = count + previous_count
        new_estimation = (avg * count + previous_avg * previous_count) / new_count

        lb, ub = hoeffding_bounds(new_estimation, new_count, self.p)

        return Score(0, new_estimation, lb, ub, new_count)

    def reduce_uncertainty_to(self, estimation: Score, width: float):
        return super().reduce_uncertainty_to(estimation, width)

    def is_estimation(self):
        return True


class SizeMetric(Metric):

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, X, y, structure: Structure, previous_estimation: Score = None) -> Score:
        bins = structure.bins
        return Score(np.sum(bins[:, 1] - bins[:, 0], axis=-1), 0, 0, 0, 0)

    def reduce_uncertainty_to(self, estimation: Score, width: float):
        return 0

    def is_estimation(self):
        return False


def is_probably_better(best_score: Score, worst_score: Score, tolerance: float):
    return worst_score.ub - best_score.lb < tolerance


def is_probably_higher(score: Score, value: float, tolerance: float):
    return value - score.lb < tolerance


def is_probably_lower(score: Score, value: float, tolerance: float):
    return value - score.ub > tolerance
