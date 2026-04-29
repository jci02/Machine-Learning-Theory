import numpy as np # for arrays

class RidgeRegGD:
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

        # add intercept column if requested
        if self.intercept:
            X = np.c_[np.ones((X.shape[0],1)), X]

        n_samples, n_features = X.shape

        # initialize theta
        theta = np.random.uniform(0,1,n_features)
        self.theta_history = [theta.copy()]

        for _ in range(self.max_iter):

            # residual part
            grad_ls = -(2/n_samples) * X.T @ (y - X @ theta) # gradient

            # ridge penalty part
            theta_tilde = theta.copy()
            if self.intercept:
                theta_tilde[0] = 0 # do not penalize intercept

            grad_penalty = 2 * self.lbda * theta_tilde

            grad = grad_ls + grad_penalty # full ridge gradient

            theta = theta - self.lr * grad # GD update

            self.theta_history.append(theta.copy())

            if np.linalg.norm(theta - self.theta_history[-2]) < self.tol:
                break

        # split parameters
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