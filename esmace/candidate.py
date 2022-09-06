from dataclasses import dataclass

from esmace.metric import Score, Metric
from esmace.sampler import Sampler
from esmace.structure import Structure


@dataclass(slots=True, frozen=False)
class Explanation:
    structure: Structure
    utility_score: Score
    restriction_score: Score
    sampling_data: dict = None

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Structure):
            return __o == self.structure
        return False

    def __hash__(self) -> int:
        return hash(self.structure)


def update_metrics(candidate: Explanation, n_points: int, utility: Metric, restriction: Metric, sampler: Sampler):
    X, y = sampler.sample(candidate.structure, n_points=n_points)
    candidate.utility_score = utility.calculate(X, y, candidate.structure, candidate.utility_score)
    candidate.restriction_score = restriction.calculate(X, y, candidate.structure, candidate.restriction_score)
