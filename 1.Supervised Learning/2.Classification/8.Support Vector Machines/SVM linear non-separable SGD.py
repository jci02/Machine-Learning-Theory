import numpy as np # for arrays and random numbers
import matplotlib.pyplot as plt # plotting
import plotly.express as px # plotting
import plotly.graph_objects as go # plotting
import plotly.io as pio


class SVM_SGD:
    def __init__(self, lr=0.01, C=0.1, max_iter=1000):
        if not isinstance(lr, (float, int)):
            raise ValueError("Learning rate needs to be numeric")
        if not isinstance(C, (float, int)):
            raise ValueError("Regularization strength needs to be numeric")
        if not isinstance(max_iter, int):
            raise ValueError("Max number of iterations need to be a number")

        self.weights = None
        self.bias = None
        self.lr = lr
        self.C = C
        self.max_iter = max_iter
        self.is_fitted = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != len(y):
            raise ValueError("X has incompatible shape with y")

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            idx = np.random.permutation(n_samples)

            for i in idx:
                xi = X[i]
                yi = y[i]

                margin = yi * (np.dot(self.weights, xi) + self.bias)

                # correct classification
                if margin >= 1:
                    self.weights = self.weights - self.lr * self.weights
                else:
                    self.weights = self.weights - self.lr * (
                        self.weights - self.C * yi * xi
                    )
                    self.bias = self.bias + self.lr * self.C * yi

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != len(self.weights):
            raise ValueError("X has different number of features than training data")

        return np.sign(X @ self.weights + self.bias)


if __name__ == "__main__":

    np.random.seed(1547)
    n = 75

    # Data
    X = np.random.uniform(1, 10, (n, 3))
    y = np.ones(n)
    y[:n // 2] = -1
    X[:n // 2] += 4  # shift class -1


    # Train model
    model = SVM_SGD(lr=0.01, C=3, max_iter=1000)
    model.fit(X, y)

    w, b = model.weights, model.bias
    pred = model.predict(X)

    print("w:", w)
    print("b:", b)
    print("accuracy:", np.mean(pred == y))


    # Grid for surfaces
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    eps = 1e-9  # numerical safety

    z0 = (-w[0] * x_grid - w[1] * y_grid - b) / (w[2] + eps)
    z_pos = (-w[0] * x_grid - w[1] * y_grid - b - 1) / (w[2] + eps)
    z_neg = (-w[0] * x_grid - w[1] * y_grid - b + 1) / (w[2] + eps)

    # Plot
    pio.renderers.default = "browser"

    colors = ["Class +1" if yi == 1 else "Class -1" for yi in y]

    fig = px.scatter_3d(x=X[:, 0],y=X[:, 1],z=X[:, 2],title="Soft Margin SVM (SGD)",labels={"x": "x1", "y": "x2", "z": "x3"},
                        color=colors,color_discrete_map={"Class -1": "red", "Class +1": "green"})
    
    # Decision boundary
    fig.add_trace(go.Surface(x=x_grid,y=y_grid,z=z0,opacity=0.5,showscale=False,name="Decision boundary"))

    fig.add_trace(go.Surface(x=x_grid,y=y_grid,z=z_pos,opacity=0.3,showscale=False,colorscale=[[0, 'blue'], [1, 'blue']],name="+1 margin"))

    fig.add_trace(go.Surface(x=x_grid,y=y_grid,z=z_neg,opacity=0.3,showscale=False,colorscale=[[0, 'blue'], [1, 'blue']],name="-1 margin"))

    # Support vectors
    margins = y * (X @ w + b)
    tol = 1e-5

    support_mask = margins <= 1 + tol
    support_vectors = X[support_mask]

    fig.add_trace(go.Scatter3d(x=support_vectors[:, 0],y=support_vectors[:, 1],z=support_vectors[:, 2],mode="markers",
                               name="Support Vectors",marker=dict(size=6, color="yellow", line=dict(color="black", width=2))))
    fig.show()