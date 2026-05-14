import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


class GradientBoostingRegressorScratch:

    def __init__(self,n_estimators=100,learning_rate=0.1,max_depth=2,random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.trees = []
        self.theta = []

        self.initial_prediction = None


    # Fit Model
    def fit(self, X, y):

        n_samples = len(y)

        # Step 1:
        # Initialize with constant prediction
        # For squared loss -> optimal constant = mean(y)
        self.initial_prediction = np.mean(y)

        # Current predictions
        current_predictions = np.full(
            shape=n_samples,
            fill_value=self.initial_prediction
        )

  
        # Boosting iterations
        for m in range(self.n_estimators):


            # Step 2(A):
            # Compute pseudo residuals
            #
            # Squared loss:
            # L = 0.5 * (y - f(x))^2
            #
            # Negative gradient:
            # r = y - f(x)


            residuals = y - current_predictions

            # Step 2(B):
            # Fit regression tree to residuals

            tree = DecisionTreeRegressor(max_depth=self.max_depth,random_state=self.random_state)

            tree.fit(X, residuals)

            # Predict residuals
            tree_predictions = tree.predict(X)

            # Step 2(C):
            # Compute terminal region values
            #
            # For squared loss:
            # optimal theta = mean residual in leaf
            #
            # sklearn trees already predict leaf means,
            # so tree_predictions already contain theta_jm


            theta = tree_predictions

            # Step 2(D):
            # Update ensemble predictions


            current_predictions += (self.learning_rate * theta)

            # Store tree
            self.trees.append(tree)

            # Track training loss
            mse = mean_squared_error(y, current_predictions)

            print(f"Iteration {m+1}")
            print(f"MSE: {mse:.4f}")
            print("-" * 40)

 
    # Predict
    def predict(self, X):

        # Start with initial prediction
        predictions = np.full(shape=X.shape[0],fill_value=self.initial_prediction)

        # Add contributions from all trees
        for tree in self.trees:

            predictions += (self.learning_rate * tree.predict(X))

        return predictions


if __name__ == "__main__":

    # Generate Regression Data
    X, y = make_regression(n_samples=200,n_features=1,noise=15,random_state=1415)

    # Sort for cleaner plotting
    sorted_idx = np.argsort(X[:, 0])

    X = X[sorted_idx]
    y = y[sorted_idx]


    # Train Gradient Boosting Model

    model = GradientBoostingRegressorScratch(n_estimators=100,learning_rate=0.1,max_depth=2,random_state=1415)

    model.fit(X, y)



    # Predictions
    predictions = model.predict(X)



    # Evaluate Model
    mse = mean_squared_error(y, predictions)

    print("\nFinal Performance")
    print(f"MSE: {mse:.4f}")



    # Plot Results
    plt.figure(figsize=(10, 6))

    # True data
    plt.scatter(X,y,color="green",label="True Data")

    # Predictions
    plt.plot(X,predictions,color="red",linewidth=2,label="Gradient Boosting Prediction")

    plt.xlabel("Feature")
    plt.ylabel("Target")

    plt.title("Gradient Boosting with Regression Trees")

    plt.legend()

    plt.grid(True)

    plt.show()