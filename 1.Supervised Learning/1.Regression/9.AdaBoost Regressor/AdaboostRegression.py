import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression # make data for Regression
import matplotlib.pyplot as plt


class AdaBoostR2:
    def __init__(self,n_estimators=20,learning_rate=0.5,max_depth=1,random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.models = []
        self.alphas = []

    def fit(self, X, y):

        n = len(y)

        # Initialize sample weights
        weights = np.ones(n) / n

        rng = np.random.default_rng(self.random_state)

        for m in range(self.n_estimators):

            # Bootstrap sampling using current weights
            bootstrap_indices = rng.choice(np.arange(n),size=n,replace=True,p=weights)

            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]

            # Train weak learner
            tree = DecisionTreeRegressor(max_depth=self.max_depth,random_state=self.random_state)

            tree.fit(X_boot, y_boot)

            # Predict on original dataset
            y_pred = tree.predict(X)

            # Absolute errors
            errors = np.abs(y - y_pred)

            # Normalize errors
            eps = 1e-10
            normalized_errors = errors / (np.max(errors) + eps)

            # Weighted error
            weighted_error = np.sum(weights * normalized_errors)

            # Avoid division problems
            weighted_error = np.clip(weighted_error,1e-10,1 - 1e-10)

            # Compute beta
            beta = weighted_error / (1 - weighted_error)

            # Learner weight
            alpha = self.learning_rate * np.log(1 / beta)

            # Update sample weights
            weights = weights * (beta ** ((1 - normalized_errors) * self.learning_rate))

            # Normalize weights
            weights = weights / np.sum(weights)

            # Store learner
            self.models.append(tree)
            self.alphas.append(alpha)

            print(f"Round {m+1}")
            print(f"Weighted Error: {weighted_error:.4f}")
            print(f"Alpha: {alpha:.4f}")
            print("-" * 40)

        self.alphas = np.array(self.alphas)

    def _weighted_median_prediction(self, predictions, weights): # _ in _weighted_median_prediction() shows that this function
                                                                 # is not supposed to be accessed publicly 

        sorted_idx = np.argsort(predictions)

        preds_sorted = predictions[sorted_idx]
        weights_sorted = weights[sorted_idx]

        cumulative_weights = np.cumsum(weights_sorted)

        cutoff = 0.5 * np.sum(weights_sorted)

        median_idx = np.where(cumulative_weights >= cutoff)[0][0]

        return preds_sorted[median_idx]

    def predict(self, X):

        n_samples = X.shape[0]

        predictions = []

        for i in range(n_samples):

            learner_predictions = np.array([model.predict(X[i].reshape(1, -1))[0]for model in self.models])

            final_prediction = self._weighted_median_prediction(learner_predictions,self.alphas)

            predictions.append(final_prediction)

        return np.array(predictions)

    def score(self, X, y):

        predictions = self.predict(X)

        mae = mean_absolute_error(y, predictions)

        return mae
    
if __name__ == "__main__":
    # Generate regression data
    X, y = make_regression(n_samples=200,n_features=1,noise=15,random_state=1500)

    # Sort data for cleaner plotting
    sorted_idx = np.argsort(X[:, 0])

    X = X[sorted_idx]
    y = y[sorted_idx]

    # Create model
    model = AdaBoostR2(n_estimators=20,learning_rate=0.5,max_depth=1,random_state=1500)

    # Train model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate model
    mae = model.score(X, y)

    print("\nFinal Performance")
    print(f"MAE: {mae:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))

    # Original data
    plt.scatter(X,y,color="green",label="True Data")

    # Model predictions
    plt.plot(X,predictions,color="red",linewidth=2,label="AdaBoost.R2 Predictions")

    plt.xlabel("Feature")
    plt.ylabel("Target")

    plt.title("AdaBoost.R2 Regression")
    plt.legend()
    plt.grid(True)
    plt.show()