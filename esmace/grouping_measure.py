import numpy as np


class GroupingMeasure:

    def calculate(self, y: np.ndarray) -> np.ndarray:
        pass


class SimpleMatchingGroupingMeasure(GroupingMeasure):

    def __init__(self, counterfactual_class) -> None:
        super().__init__()
        self.counterfactual_class = counterfactual_class

    def calculate(self, y: np.ndarray) -> np.ndarray:
        return (y == self.counterfactual_class).astype(int)
