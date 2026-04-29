import numpy as np # for arrays
import plotly.express as px # plotting
import plotly.io as pio # set where plots are shown

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None # eigenvectors
        self.explained_variance_ = None # eigenvalues

    def fit(self, X):
        X = np.array(X)

        # center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store top components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]

        return self

    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
import plotly.express as px
import plotly.graph_objects as go

if __name__ == "__main__":

    np.random.seed(1239)
    n = 200

    # Generate 3D correlated data
    mean = [3, 4, 5]
    cov = [[6, 3, 2],
           [3, 2, 1],
           [2, 1, 2]]

    X = np.random.multivariate_normal(mean, cov, size=n)

    # Plot original 3D data
    pio.renderers.default = "browser" # to show plot in browser 
    fig = px.scatter_3d(x=X[:, 0],y=X[:, 1],z=X[:, 2],title="Original 3D Data")
    fig.show()

    # Apply PCA -> 2D
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X)

    print("Explained variance (eigenvalues):", pca.explained_variance_)
    print("Principal components:\n", pca.components_)

    # Plot projected 2D data
    fig2 = px.scatter(x=X_proj[:, 0],y=X_proj[:, 1],title="Projected Data (2D PCA)")

    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    fig2.add_vline(x=0, line_dash="dash", line_color="black")

    fig2.show()