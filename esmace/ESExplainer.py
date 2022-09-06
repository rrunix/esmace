from dataclasses import dataclass
from typing import List

from tqdm.auto import tqdm

from esmace.candidate import Explanation, update_metrics
from esmace.discretizer import Discretizer
from esmace.expand_strategy import ExpandStrategy
from esmace.metric import Metric, is_probably_higher, is_probably_lower
from esmace.neighborhood import Neighborhood
from esmace.mab import mab_lub
from esmace.sampler import Sampler
from esmace.structure import Structure
from esmace.utils import check_is_fitted


@dataclass(slots=True, frozen=True)
class Restriction:
    metric: Metric
    minimum_value: float


class ESExplainer:

    def __init__(self, sampler: Sampler, discretizer: Discretizer, expand_strategy: ExpandStrategy,
                 initial_sampling_size=100) -> None:
        self.sampler = sampler
        self.discretizer = discretizer
        self.initial_sampling_size = initial_sampling_size
        self.expand_strategy = expand_strategy

    def fit(self, X, y):
        self.sampler.dataset_fit(X)
        self.discretizer.fit(X, y)
        self.X_ = X

    def explain(self, x, utility: Metric, neighborhood: Neighborhood, restriction: Restriction, tolerance: float = 0.01,
                n_iterations: int = 50, beam_size: int = 5):
        check_is_fitted(self)
        self.discretizer.fit_target_sample(x)
        self.sampler.fit_discretizer(self.discretizer)
        self.expand_strategy.fit_discretizer(self.discretizer, neighborhood)

        self.tolerance_ = tolerance
        self.utility_ = utility
        self.neighborhood_ = neighborhood
        self.restriction_ = restriction

        return self._generate_explanation(x, n_iterations, beam_size)

    def _select_k_best(self, candidates, k):
        valid, invalid = self._filter_candidates(candidates)
        k_best = self._mab(valid, k)

        if len(k_best) < k:
            remaining_k = k - len(k_best)
            surviving_invalid = self._mab(invalid, remaining_k, scorer='restriction')
            k_best += surviving_invalid

        return k_best

    def _generate_explanation(self, x, n_iterations, beam_size):
        structure = self.discretizer.to_structure(x)
        factual_candidate = self._create_candidate(structure)

        candidates = [factual_candidate]
        previously_seen_structures = {factual_candidate.structure}
        hall_of_fame = []

        for _ in tqdm(range(n_iterations)):
            surviving_candidates = self._select_k_best(candidates, beam_size)

            only_old_candidates = True
            new_candidates = []
            for candidate in surviving_candidates:
                new_offspring = set(self._expand_candidate(candidate, previously_seen_structures))

                if len(new_offspring) > 0:
                    only_old_candidates = False
                    new_candidates.extend(new_offspring)
                    previously_seen_structures |= {new_candidate.structure for new_candidate in new_offspring}

            # surviving_candidates: 
            hall_of_fame = self._select_k_best(hall_of_fame + surviving_candidates, beam_size)

            candidates = new_candidates
            if only_old_candidates:
                break

        # Keep best
        best = self._select_k_best(hall_of_fame, 1)
        return best

    def _expand_candidate(self, candidate, previously_seen_structures):
        structures = self.expand_strategy.expand(candidate.structure)
        return [self._create_candidate(structure) for structure in structures if
                structure not in previously_seen_structures]

    def _mab(self, candidates: List[Explanation], m: int, scorer: str = 'utility'):
        if scorer == 'utility':
            metric = self.utility_
            scorer = lambda x: x.utility_score
        elif scorer == 'restriction':
            metric = self.restriction_.metric
            scorer = lambda x: x.restriction_score

        if metric.is_estimation():
            return mab_lub(candidates, m, self.tolerance_, scorer, self._update_metrics)
        else:
            return sorted(candidates, key=lambda x: scorer(x).score, reverse=True)[:m]

    def _filter_candidates(self, candidates: List[Explanation]):
        valid, invalid = [], []

        for candidate in candidates:
            if self._validate_candidate(candidate):
                valid.append(candidate)
            else:
                invalid.append(candidate)

        return valid, invalid

    def _create_candidate(self, structure: Structure) -> Explanation:
        X, y = self.sampler.initial_sampling(structure, n_points=self.initial_sampling_size)
        utility_score = self.utility_.calculate(X, y, structure, None)
        restriction_score = self.restriction_.metric.calculate(X, y, structure, None)
        return Explanation(structure, utility_score, restriction_score, None)

    def _validate_candidate(self, candidate: Explanation):
        restriction_metric = self.restriction_.metric
        mininum_value = self.restriction_.minimum_value

        while True:
            if is_probably_higher(candidate.restriction_score, mininum_value, self.tolerance_):
                return True
            elif is_probably_lower(candidate.restriction_score, mininum_value, self.tolerance_):
                return False
            else:
                if candidate.restriction_score.score > mininum_value:
                    width = candidate.restriction_score.score - mininum_value
                else:
                    width = mininum_value - candidate.restriction_score.score
                n_points = restriction_metric.reduce_uncertainty_to(candidate.restriction_score, width)
                self._update_metrics(candidate, n_points)

    def _update_metrics(self, candidate: Explanation, n_points: int):
        update_metrics(candidate, n_points, self.utility_, self.restriction_.metric, self.sampler)
