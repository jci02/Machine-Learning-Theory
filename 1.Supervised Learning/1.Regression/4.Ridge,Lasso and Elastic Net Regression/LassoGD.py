import numpy as np # for arrays
from sklearn.linear_model import Lasso # for Ridge Regression
import matplotlib.pyplot as plt # for plotting
from sklearn.metrics import mean_squared_error # for mse
from sklearn.datasets import make_regression # create regression toy data
from sklearn.model_selection import train_test_split # for splitting data in train and test set

class LassoRegGD:
    def __init__(self, intercept=True, lbda=0.0, lr=0.001, max_iter=5000, tol=1e-6):
        if not isinstance(intercept, bool):
            raise ValueError("intercept must be boolean")
        if not isinstance(lbda, (int,float)):
            raise ValueError("lambda must be numeric")
        if not isinstance(lr, (int,float)):
            raise ValueError("learning rate must be numeric")
        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be integer")

        self.intercept = intercept
        self.lbda = lbda
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

        self.bias = None
        self.weights = None
        self.theta_history = []
        self.is_fitted = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1,1)

        if X.shape[0] != len(y):
            raise ValueError("X has incompatible shape with y")

        # add intercept
        if self.intercept:
            X = np.c_[np.ones((X.shape[0],1)), X]

        n_samples, n_features = X.shape

        theta = np.random.uniform(0,1,n_features)
        self.theta_history = [theta.copy()]

        for _ in range(self.max_iter):

            # least squares gradient
            grad_ls = -(2/n_samples) * X.T @ (y - X @ theta)

            # lasso subgradient
            theta_tilde = theta.copy()
            if self.intercept:
                theta_tilde[0] = 0   # do not penalize intercept

            grad_l1 = self.lbda * np.sign(theta_tilde)

            # full subgradient
            grad = grad_ls + grad_l1

            # update
            theta = theta - self.lr * grad

            self.theta_history.append(theta.copy())

            if np.linalg.norm(theta - self.theta_history[-2]) < self.tol:
                break

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

        if X.shape[1] != len(self.weights):
            raise ValueError("X has different number of features than training data")

        return X @ self.weights + self.bias
    
if __name__ == "__main__":
    # Generate data (use more features to see sparsity effect)
    X, y = make_regression(n_samples=100,n_features=5,n_informative=2,noise=10,random_state=42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lambdas = [0.1, 1, 10, 100]

    print("=== Lasso: Coefficients + MSE Comparison ===\n")

    for lbda in lambdas:
        # Our model
        model = LassoRegGD(intercept=True, lbda=lbda, lr=0.001, max_iter=10000)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)

        # Sklearn model
        sk_model = Lasso(alpha=lbda, fit_intercept=True, max_iter=10000)
        sk_model.fit(X_train, y_train)

        sk_y_pred_test = sk_model.predict(X_test)
        sk_mse = mean_squared_error(y_test, sk_y_pred_test)

        print(f"Lambda = {lbda}")
        print(f"Our weights:      {np.round(model.weights, 4)}")
        print(f"Sklearn weights: {np.round(sk_model.coef_, 4)}")
        print(f"Our bias: {model.bias:.4f} | Sklearn bias: {sk_model.intercept_:.4f}")
        print(f"Our MSE: {mse:.2f} | Sklearn MSE: {sk_mse:.2f}")
        print("-" * 60)

    # Optional visualization (1D case for plotting) 
    # regenerate simple 1D data for plotting
    X_1d, y_1d = make_regression(n_samples=100, n_features=1, noise=15, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X_1d, y_1d, test_size=0.2, random_state=0)

    X_plot = np.linspace(X_1d.min(), X_1d.max(), 200).reshape(-1, 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.6, label="Train data")
    plt.scatter(X_test, y_test, alpha=0.6, label="Test data")

    for lbda in lambdas:
        model = LassoRegGD(intercept=True, lbda=lbda, lr=0.001, max_iter=10000)
        model.fit(X_train, y_train)

        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f"λ={lbda}")

    plt.title("Lasso Regression (GD)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()