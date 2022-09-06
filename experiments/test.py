import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
y %= 2

clf = DecisionTreeClassifier(random_state=0, max_depth=4).fit(X, y)

from esmace.discretizer import TabularDiscretizer
from esmace.sampler import TabularSampler, CachingTabularSampler
from esmace.expand_strategy import StepExpandStrategy
from esmace.neighborhood import NoNeighborhood
from esmace.metric import FidelityMetric, SizeMetric
from esmace.grouping_measure import SimpleMatchingGroupingMeasure
from esmace.ESExplainer import ESExplainer, Restriction

discretizer = TabularDiscretizer(num_bins=10)
sampler = CachingTabularSampler(lambda x: np.ravel(clf.predict(x)), n_points_cache=100_000)
expand = StepExpandStrategy(max_step=1)

instance = X[25]
label = clf.predict(instance.reshape(1, -1))
fidelity = FidelityMetric(SimpleMatchingGroupingMeasure(label), p=0.01)

explainer = ESExplainer(sampler, discretizer, expand, initial_sampling_size=None)
explainer.fit(X, y)
exp = explainer.explain(X[0], SizeMetric(), NoNeighborhood(), Restriction(fidelity, 0.95), beam_size=10, n_iterations=100)
print(exp)
