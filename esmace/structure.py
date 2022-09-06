from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Structure:
    bins: np.ndarray

    def __eq__(self, __o: object) -> bool:
        return __o is not None and np.all(__o.bins == self.bins)

    def __hash__(self) -> int:
        return hash(tuple(self.bins.ravel()))
