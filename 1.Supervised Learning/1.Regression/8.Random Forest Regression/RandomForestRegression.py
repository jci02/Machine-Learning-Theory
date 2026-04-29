import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10):
        if not isinstance(n_trees, int):
            raise ValueError("n_trees must be integer")

        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        self.trees = []
        self.is_fitted = False


    # Bootstrap sample helper
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]

        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs], y[idxs]



    # Fit forest
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.trees = []

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split,max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        self.is_fitted = True
        return self


    # Predict
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        # collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # average over rows
        return np.mean(tree_preds, axis=0)