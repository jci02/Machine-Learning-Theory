import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


class GradientBoostingBinaryClassifierScratch:

    def __init__(self,n_estimators=100,learning_rate=0.1,max_depth=2,random_state=None):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.trees = []

        # Initial log-odds prediction
        self.initial_prediction = None


    # Sigmoid Function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    # Fit Model
    def fit(self, X, y):

        n_samples = len(y)


        # Step 1:
        # Initialize model
        #
        # For logistic loss:
        #
        # f0 = log(p / (1-p))

        p = np.mean(y)

        eps = 1e-10

        self.initial_prediction = np.log((p + eps) / (1 - p + eps))

        # Initial scores
        current_scores = np.full(shape=n_samples,fill_value=self.initial_prediction)


        # Boosting Loop
        for m in range(self.n_estimators):


            # Step 2(A):
            # Compute probabilities

            probabilities = self.sigmoid(current_scores)


            # Step 2(B):
            # Compute pseudo-residuals
            #
            # Logistic loss gradient:
            #
            # residual = y - p(x)

            residuals = y - probabilities

            # Step 2(C):
            # Fit regression tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth,random_state=self.random_state)

            tree.fit(X, residuals)

            # Tree predictions
            gamma = tree.predict(X)


            # Step 2(D):
            # Update scores
            current_scores += (self.learning_rate * gamma)

            # Store tree
            self.trees.append(tree)


            # Monitor training accuracy
            current_probabilities = self.sigmoid(current_scores)

            predictions = (current_probabilities >= 0.5).astype(int)

            accuracy = accuracy_score(y, predictions)

            print(f"Iteration {m+1}")
            print(f"Accuracy: {accuracy:.4f}")
            print("-" * 40)


    # Predict Probabilities
    def predict_proba(self, X):

        scores = np.full(shape=X.shape[0],fill_value=self.initial_prediction)

        # Add contributions from trees
        for tree in self.trees:
            scores += (self.learning_rate *tree.predict(X))

        probabilities = self.sigmoid(scores)

        return probabilities


    # Predict Classes
    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)


if __name__ == "__main__":
    # Generate Binary Classification Data
    X, y = make_classification(n_samples=300,n_features=2,n_redundant=0,n_clusters_per_class=1,class_sep=2,random_state=1415)



    # Train Gradient Boosting Classifier
    model = GradientBoostingBinaryClassifierScratch(n_estimators=100,learning_rate=0.1,max_depth=2,random_state=1415)
    model.fit(X, y)


    # Predictions
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    print("\nFinal Performance")
    print(f"Accuracy: {accuracy:.4f}")


    # Plot Decision Boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)

    Z = Z.reshape(xx.shape)


    plt.figure(figsize=(10, 6))

    # Decision boundary
    plt.contourf(xx,yy,Z,alpha=0.3)

    # Data points
    plt.scatter(X[:, 0],X[:, 1],c=y,edgecolor="k")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.title("Gradient Boosting Binary Classification")
    plt.grid(True)
    plt.show()