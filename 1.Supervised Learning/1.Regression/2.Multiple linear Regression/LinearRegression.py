import numpy as np # for random numbers and arrays

class LinReg:
    def __init__(self, intercept=True, copy_X=True):
        if not isinstance(intercept, bool) or not isinstance(copy_X, bool):
            raise ValueError("intercept and copy_X must be booleans")

        self.intercept = intercept # indicator wether intercept cplumn should be added to X
        self.copy_X = copy_X # indicator wether original X (without) intercept schould be stored
        self.weights = None
        self.bias = None # bias only if intercept is True
        self.is_fitted = False 
        self.X_original = None # stores original X

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1,1) # reshape to matrix if X only has 1 dimension

        if X.shape[0] != len(y): # number of rows must be equal to length of y
            raise ValueError("X has incompatible shape with y")

        if self.intercept and self.copy_X:
            self.X_original = X.copy() # store original 

        if self.intercept: # if only intercept is true
            X = np.c_[np.ones((X.shape[0],1)), X]

        theta = np.linalg.pinv(X) @ y # same as np.linalg.inv(X.T @ X) @ X.T @ y but more numerically stable

        if self.intercept:
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.bias = 0 # no bias
            self.weights = theta

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1,1) # reshape to matrix if X only has 1 dimension

        if X.shape[1] != len(self.weights): # number of columns in X should be equal to length of weights array
            raise ValueError("X has different number of features than training data")

        return X @ self.weights + self.bias


def mse(y, predictions): # mean squarred error
    y = np.array(y)
    predictions = np.array(predictions)
    return np.round(np.mean((y - predictions)**2),4)