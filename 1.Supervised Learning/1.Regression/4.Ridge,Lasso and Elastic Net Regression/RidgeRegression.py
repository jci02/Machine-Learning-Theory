import numpy as np # for arrays

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

        theta = np.linalg.inv(X.T @ X + self.lbda * I) @ X.T @ y # REidge solution

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