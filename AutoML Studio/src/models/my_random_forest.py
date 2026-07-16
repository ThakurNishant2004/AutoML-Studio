import numpy as np
from collections import Counter
from my_decision_tree import MyDecisionTree, Node

class MyRandomForest:

    def __init__(
        self,
        n_estimators=10,
        max_depth=10,
        min_samples_split=2
    ):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.trees = []

    def bootstrap_sample(self, X, y):

        n_samples = X.shape[0]

        indices = np.random.choice(
            n_samples,
            n_samples,
            replace=True
        )

        return X[indices], y[indices]

    def fit(self, X, y):

        self.trees = []

        for _ in range(self.n_estimators):

            X_sample, y_sample = self.bootstrap_sample(X, y)

            tree = MyDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        return self

    def predict(self, X):

        predictions = np.array(
            [tree.predict(X) for tree in self.trees]
        )

        # Shape:
        # (n_estimators, n_samples)

        predictions = predictions.T

        final_predictions = []

        for pred in predictions:

            vote = Counter(pred).most_common(1)[0][0]

            final_predictions.append(vote)

        return np.array(final_predictions)