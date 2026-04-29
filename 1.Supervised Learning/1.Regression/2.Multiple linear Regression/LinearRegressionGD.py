import numpy as np

def gradient(X, y, theta):
    n = len(y)
    return -(2/n) * X.T @ (y - X @ theta) # smaller step size than -2 * X.T @ (y - X @ theta) but still same


class LinRegGD:
    def __init__(self, intercept=True, lr=0.001, max_iter=5000, tol=1e-6):
        if not isinstance(intercept, bool) or not isinstance(lr, (int,float)) or not isinstance(max_iter, int):
            raise ValueError("Incompatible types of parameters")

        self.intercept = intercept # if intercept = True that means model will adds intercept
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

        self.weights = None
        self.bias = None
        self.theta_history = []
        self.is_fitted = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1,1)

        if X.shape[0] != len(y):
            raise ValueError("X has incompatible shape with y")

        # Add intercept column
        if self.intercept:
            X = np.c_[np.ones((X.shape[0],1)), X]

        n_samples, n_features = X.shape

        # initialize theta randomly
        theta = np.random.uniform(0,1,n_features)

        self.theta_history = [theta.copy()]

        for _ in range(self.max_iter):
            slope = gradient(X, y, theta)
            theta = theta - self.lr * slope
            self.theta_history.append(theta.copy())

            if np.linalg.norm(theta - self.theta_history[-2]) < self.tol:
                break

        # split theta into bias + weights
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