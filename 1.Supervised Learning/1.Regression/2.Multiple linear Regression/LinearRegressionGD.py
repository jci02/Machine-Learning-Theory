import numpy as np

class LinRegGD:
    def __init__(self,intercept):
        if not isinstance(intercept, bool):
            raise ValueError("Intercept must be boolean")
        self.intercept = intercept
        self.bias = None
        self.weights = None
        self.is_fitted = False

    def fit(X,y):
        X = np.array(X)
        y = np.array(y)
