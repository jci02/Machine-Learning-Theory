import numpy as np
from matplotlib import pyplot as plt


class KNNRegressor:
    def __init__(self, k=5, distance="euclidean", weighted=False):
        if distance not in ["euclidean", "manhattan"]:
            raise ValueError("distance must be 'euclidean' or 'manhattan'")
        
        self.k = k
        self.distance = distance
        self.weighted = weighted

        self.X_train = None
        self.y_train = None
        self.is_fitted = False

    def _compute_distance(self, x1, x2):
        if self.distance == "euclidean":
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.distance == "manhattan":
            return np.sum(np.abs(x1 - x2))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")

        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []

        for x in X:
            # compute all distances
            distances = [(self._compute_distance(x, x_train), idx)for idx, x_train in enumerate(self.X_train)]

            # sort by distance
            distances_sorted = sorted(distances, key=lambda t: t[0])

            # select k nearest
            neighbors = distances_sorted[:self.k]

            neighbor_indices = [idx for (_, idx) in neighbors]
            neighbor_distances = np.array([dist for (dist, _) in neighbors])
            neighbor_values = self.y_train[neighbor_indices]

            if self.weighted:
                weights = 1 / (neighbor_distances + 1e-9)
                pred = np.sum(weights * neighbor_values) / np.sum(weights)
            else:
                pred = np.mean(neighbor_values)

            predictions.append(pred)

        return np.array(predictions)
    

    def get_neighbors(self, x):
        """Return k nearest neighbors (points + distances) for a single query point"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        x = np.array(x)

        distances = [
            (self._compute_distance(x, x_train), idx)
            for idx, x_train in enumerate(self.X_train)
        ]

        distances_sorted = sorted(distances, key=lambda t: t[0])
        neighbors = distances_sorted[:self.k]

        neighbor_indices = [idx for (_, idx) in neighbors]
        neighbor_distances = np.array([dist for (dist, _) in neighbors])

        return self.X_train[neighbor_indices], self.y_train[neighbor_indices], neighbor_distances
    
if __name__ == "__main__":
    # Generate data
    np.random.seed(1246)
    n = 100
    X = np.random.uniform(0, 10, (n, 1))
    X = np.hstack((X, 4.5 * X + 1.5 * np.random.uniform(10, 20, (n, 1))))

    X_input = X[:, 0].reshape(-1, 1)
    y = X[:, 1]

    new_x = np.array([[6]])

    # Manhattan
    knn_m = KNNRegressor(k=5, distance="manhattan", weighted=False)
    knn_m.fit(X_input, y)

    neighbors_X_m, neighbors_y_m, dist_m = knn_m.get_neighbors(new_x[0])
    prediction = np.mean(neighbors_y_m)

    weights_m = 1 / (dist_m + 1e-9)
    weighted_prediction = np.sum(weights_m * neighbors_y_m) / np.sum(weights_m)

    # Euclidean
    knn_e = KNNRegressor(k=10, distance="euclidean", weighted=False)
    knn_e.fit(X_input, y)

    neighbors_X_e, neighbors_y_e, dist_e = knn_e.get_neighbors(new_x[0])
    prediction_e = np.mean(neighbors_y_e)

    weights_e = 1 / (dist_e + 1e-9)
    weighted_prediction_e = np.sum(weights_e * neighbors_y_e) / np.sum(weights_e)

    # Plot
    plt.figure(figsize=(12,5))

    # Manhattan plot
    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], color="green", label="Data")
    plt.scatter(new_x[0], prediction, s=80, color="red", label="Standard")
    plt.scatter(new_x[0], weighted_prediction, s=80, color="orange", label="Weighted")
    plt.axvline(new_x[0], linestyle="--", color="gray")
    plt.scatter(neighbors_X_m[:,0], neighbors_y_m, label="Neighbors")
    plt.title("5-NN (Manhattan)")
    plt.legend()

    # Euclidean plot
    plt.subplot(1,2,2)
    plt.scatter(X[:,0], X[:,1], color="green", label="Data")
    plt.scatter(new_x[0], prediction_e, s=80, color="red", label="Standard")
    plt.scatter(new_x[0], weighted_prediction_e, s=80, color="orange", label="Weighted")
    plt.axvline(new_x[0], linestyle="--", color="gray")
    plt.scatter(neighbors_X_e[:,0], neighbors_y_e, label="Neighbors")
    plt.title("10-NN (Euclidean)")
    plt.legend()

    plt.show()