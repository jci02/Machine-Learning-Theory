import numpy as np # for arrays
from sklearn.linear_model import Ridge # for Ridge Regression
import matplotlib.pyplot as plt # for plotting
from sklearn.metrics import mean_squared_error # for mse
from sklearn.datasets import make_regression # create regression toy data
from sklearn.model_selection import train_test_split # for splitting data in train and test set

def standardize(X): # standardize features
    feature_means = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)
    feature_std[feature_std == 0] = 1
    return (X - feature_means) / feature_std, feature_means, feature_std


class RidgeReg:
    def __init__(self, intercept=True, lbda=0, standardize_features=False):
        if not isinstance(intercept, bool):
            raise ValueError("Intercept must be boolean")
        if not isinstance(lbda, (int,float)):
            raise ValueError("Lambda must be numeric")

        self.intercept = intercept # if intercept = True that means we add intercept column
        self.lbda = lbda # lambda value
        self.standardize_features = standardize_features # indicator if we first standardize

        self.is_fitted = False
        self.bias = None
        self.weights = None

        self.feature_means = None # mean per feature / column of X
        self.feature_std = None # standard deviation per feature / column of X

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1,1) # reshape to (n,1) if 1 dimensional with n beign sample size

        if X.shape[0] != len(y): # number of rows of X needs to math length of y
            raise ValueError("X has incompatible shape with y") 

        if self.standardize_features: # standardize first
            X, self.feature_means, self.feature_std = standardize(X)

        if self.intercept: # then add intercept
            X = np.c_[np.ones((X.shape[0],1)), X]

        I = np.eye(X.shape[1]) # diagonal matrix

        if self.intercept:
            I[0,0] = 0 # do not penalize intercept

        theta = np.linalg.solve(X.T @ X + self.lbda * I, X.T @ y) # same as np.linalg.inv(X.T @ X + self.lbda * I) @ X.T @ y but better

        if self.intercept:
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.bias = 0
            self.weights = theta

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1,1)

        if self.standardize_features:
            X = (X - self.feature_means) / self.feature_std # also standardize test set

        if X.shape[1] != len(self.weights):
            raise ValueError("X has different number of features than training data")

        return X @ self.weights + self.bias
    

# Testing
# if __name__ == "__main__": controls whether a Python file runs as a script or is imported as a module
if __name__ == "__main__": # __name__ == "__main__" means only run this code if this file is executed directly, not imported
    # Generate data
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=1140)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1141)

    # Use FULL range for smooth plotting
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    lambdas = [0, 1, 10, 100]

    plt.figure(figsize=(10, 6))

    # Plot both train and test data
    plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Train data")
    plt.scatter(X_test, y_test, color="red", alpha=0.6, label="Test data")

    print("Coefficient + MSE Comparison\n")

    for lbda in lambdas:
        # Our model
        model = RidgeReg(intercept=True, lbda=lbda, standardize_features=False)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_plot = model.predict(X_plot)

        mse = mean_squared_error(y_test, y_pred_test)

        # Sklearn model
        sk_model = Ridge(alpha=lbda, fit_intercept=True)
        sk_model.fit(X_train, y_train)

        sk_y_pred_test = sk_model.predict(X_test)
        sk_mse = mean_squared_error(y_test, sk_y_pred_test)

        # Plot regression line
        plt.plot(X_plot, y_plot, label=f"λ={lbda}")

        # Print comparison
        print(f"Lambda = {lbda}")
        print(f"Our weights: {model.weights}, bias: {model.bias}")
        print(f"Sklearn weights: {sk_model.coef_}, bias: {sk_model.intercept_}")
        print(f"Our MSE: {mse:.2f} | Sklearn MSE: {sk_mse:.2f}")
        print("-" * 60)

    plt.title("Ridge Regression (Closed Form)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()