import numpy as np # random numbers and arrays

class LinReg:
    def __init__(self, intercept=True, copy_X=True):
        if not isinstance(intercept, bool) or not isinstance(copy_X, bool):
            raise ValueError("intercept and copy_X must be booleans")

        self.intercept = intercept # intercept column
        self.copy_X = copy_X # keep original X without intercept column if intercept=True
        self.weights = None 
        self.bias = None
        self.is_fitted = False # indicator if already fitted
        self.X_original = None # store original X if intercept = True

    def fit(self, X, y):
        if self.intercept and self.copy_X: # if intercept and copy_X are true 
            self.X_original = X.copy() # store original

        if self.intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X] # add intercept column

        theta = np.linalg.pinv(X) @ y # same as np.linalg.inv(X.T @ X) @ X.T @ y but more stable

        if self.intercept:
            self.bias = theta[0] # intercept term
            self.weights = theta[1:] # feature weights
        else:
            self.weights = theta # feature weights

        self.is_fitted = True # update fitted status

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Model not fitted yet") # only predict if model fitted

        X = np.array(X)

        return X @ self.weights + self.bias
    
def mse(y, predictions):
    return np.round(np.mean((y-predictions)**2),4)




