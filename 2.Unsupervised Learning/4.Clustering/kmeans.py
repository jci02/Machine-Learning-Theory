import numpy as np # for arrays 
import matplotlib.pyplot as plt # plotting

class KMeans:
    def __init__(self, K=3, max_iter=100, tol=1e-6, metric="euclidean", seed=0):
        if metric not in ["euclidean", "manhattan"]:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")

        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.seed = seed

        self.means = None
        self.labels = None
        self.history = []


    # distance functions
    def _euclidean(self, X, means):
        return np.linalg.norm(X[:, None, :] - means[None, :, :], axis=2)

    def _manhattan(self, X, means):
        return np.sum(np.abs(X[:, None, :] - means[None, :, :]), axis=2)

    def _get_distances(self, X, means):
        if self.metric == "euclidean":
            return self._euclidean(X, means)
        else:
            return self._manhattan(X, means)

    # fit / predict
    def fit(self, X):
        X = np.array(X)
        N, D = X.shape

        np.random.seed(self.seed)
        idx = np.random.choice(N, self.K, replace=False) # get K random number between 1 and N
        means = X[idx] # get K X.shape[1] dimensional vectors (class means)

        self.history = []

        for _ in range(self.max_iter):

            # assignment step
            distances = self._get_distances(X, means) # distances of every one of the K mean vectors to all data points in dimensions (N, K)
            labels = np.argmin(distances, axis=1) # identify minimum distance in every row

            self.history.append((means.copy(), labels.copy()))

            # update step
            new_means = np.zeros_like(means)

            for k in range(self.K):
                points = X[labels == k]

                if len(points) == 0:
                    new_means[k] = X[np.random.randint(0, N)] # reinitialize empty cluster randomly
                else:
                    new_means[k] = points.mean(axis=0)

            # convergence
            if np.linalg.norm(new_means - means) < self.tol:
                means = new_means
                break

            means = new_means

        self.means = means
        self.labels = labels
        return self

    def predict(self, X):
        X = np.array(X)
        distances = self._get_distances(X, self.means)
        return np.argmin(distances, axis=1)
    

if __name__ == "__main__":
    # create toy data
    np.random.seed(42)
    X1 = np.random.randn(100, 2) + np.array([0, 0])
    X2 = np.random.randn(100, 2) + np.array([5, 5])
    X3 = np.random.randn(100, 2) + np.array([0, 5])

    X = np.vstack((X1, X2, X3))

    # train kmeans
    kmeans = KMeans(K=3, metric="euclidean", max_iter=100)
    kmeans.fit(X)

    labels = kmeans.labels
    means = kmeans.means

    # plot
    plt.figure(figsize=(7, 6))

    for k in range(3):
        cluster = X[labels == k]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {k}")

    # centroids
    plt.scatter(means[:, 0], means[:, 1],c="black", marker="x", s=200, label="Centroids")

    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()