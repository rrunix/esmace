from typing import Callable, List

from esmace.candidate import Explanation
from esmace.metric import Score, is_probably_better


def mab_lub(candidates: List[Explanation], m: int, tolerance: float, score: Callable[[Explanation], Score],
            update_metrics: Callable[[Explanation, int], None]):
    """
    TODO: Remaining is almost sorted (only two pos can change), optimize this to avoid full sort.

    Args:
        candidates:
        m:
        tolerance:
        score:
        update_metrics:

    Returns:

    """

    selected = list()
    remaining = list(candidates)

    num_discarded = 0
    num_arms = len(candidates)

    while len(selected) < m and num_discarded < num_arms - m:

        if len(remaining) == 1:
            last = remaining.pop()
            selected.append(last)
            break

        remaining.sort(key=lambda x: score(x).lb, reverse=True)
        best_by_lb, second_best_by_lb = remaining[0], remaining[1]
        worst_by_lb, second_worst_by_lb = remaining[-1], remaining[-2]

        remaining.sort(key=lambda x: score(x).ub, reverse=True)
        best_by_ub, second_best_by_ub = remaining[0], remaining[1]
        worst_by_ub, second_worst_by_ub = remaining[-1], remaining[-2]

        compare_best = best_by_ub if best_by_ub != best_by_lb else second_best_by_ub
        compare_worst = worst_by_lb if worst_by_lb != worst_by_ub else second_worst_by_lb

        if is_probably_better(score(best_by_lb), score(compare_best), tolerance):
            first = remaining.pop(0)
            selected.append(first)
        elif is_probably_better(score(worst_by_ub), score(compare_worst), tolerance):
            num_discarded += 1
            remaining.pop()
        else:
            best_diff = score(best_by_lb).lb - score(compare_best).ub
            worst_diff = score(compare_worst).lb - score(worst_by_ub).ub

            if best_diff > worst_diff:
                reduce_bounds_diff(best_by_lb, compare_best, update_metrics, tolerance)
            else:
                reduce_bounds_diff(compare_worst, worst_by_ub, update_metrics, tolerance)

    if len(selected) == m:
        return selected
    else:
        return selected + remaining


def reduce_bounds_diff(best: Explanation, second_best: Explanation, update_metrics: Callable[[Explanation, int], None],
                       max_diff: float):
    diff = max_diff - (best.utility_score.lb - second_best.utility_score.ub)
    diff_split = diff / 2

    n_points_best = 100  # utility.reduce_uncertainty_to(best, diff_split)
    n_points_second_best = 100  # utility.reduce_uncertainty_to(second_best, diff_split)
    update_metrics(best, n_points_best)
    update_metrics(second_best, n_points_second_best)
