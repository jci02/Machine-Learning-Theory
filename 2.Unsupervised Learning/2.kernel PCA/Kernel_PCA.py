import numpy as np # for arrays
from sklearn.datasets import make_s_curve # make non linear data
import plotly.express as px # for plotting
import plotly.io as pio # set where plots are shown


class KernelPCA:
    def __init__(self, n_components=2, kernel="rbf", **kernel_params):
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params

        self.X_fit = None
        self.eigvals_ = None
        self.eigvecs_ = None
        self.K_fit_ = None

    def _centering(self, n):
        return np.eye(n) - np.ones((n, n)) / n

    def _rbf_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        X = X.T
        Y = Y.T

        sq_X = np.sum(X**2, axis=1, keepdims=True)
        sq_Y = np.sum(Y**2, axis=1, keepdims=True)

        dist_sq = sq_X + sq_Y.T - 2 * X @ Y.T

        sigma = self.kernel_params.get("sigma", 1.0)
        return np.exp(-dist_sq / (2 * sigma**2))

    def _poly_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        X = X.T
        Y = Y.T

        d = self.kernel_params.get("d", 2)
        c = self.kernel_params.get("c", 0)

        return (X @ Y.T + c) ** d

    def _compute_kernel(self, X, Y=None):
        if self.kernel == "rbf":
            return self._rbf_kernel(X, Y)
        elif self.kernel == "poly":
            return self._poly_kernel(X, Y)
        else:
            raise ValueError("Kernel must be 'rbf' or 'poly'")

    def fit(self, X):
        X = np.array(X)
        self.X_fit = X

        n = X.shape[1]

        # Gram matrix
        K = self._compute_kernel(X)

        # Center
        H = self._centering(n)
        Kc = H @ K @ H
        self.K_fit_ = Kc

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(Kc)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals = eigvals[:self.n_components]
        eigvecs = eigvecs[:, :self.n_components]

        # Normalize
        eigvals = np.maximum(eigvals, 1e-12)
        eigvecs = eigvecs / np.sqrt(eigvals)

        self.eigvals_ = eigvals
        self.eigvecs_ = eigvecs

        return self

    def transform(self, X):
        X = np.array(X)

        K = self._compute_kernel(self.X_fit, X)

        return self.eigvecs_.T @ K

    def fit_transform(self, X):
        self.fit(X)
        return self.eigvecs_.T @ self.K_fit_
    

if __name__ == "__main__":
    np.random.seed(1746)

    # Generate S-curve data
    X, t = make_s_curve(n_samples=500, noise=0.05)

    # Transpose to match your convention (D x n)
    X = X.T  # (3 x N)

    # Plot original 3D data
    pio.renderers.default = "browser" # to show plot in browser
    fig = px.scatter_3d(x=X[0],y=X[1],z=X[2],color=t,title="Original S-Curve (3D)")
    fig.show()

    # Kernel PCA
    kpca = KernelPCA(n_components=2,kernel="rbf",sigma=1.0)

    Z = kpca.fit_transform(X)  # shape (2, N)

    print("Eigenvalues:", kpca.eigvals_)

    # Plot projected data
    fig2 = px.scatter(x=Z[0],y=Z[1],color=t,title="Kernel PCA Projection (2D)")

    fig2.show()