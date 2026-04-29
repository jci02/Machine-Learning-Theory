import numpy as np
from matplotlib import pyplot as plt

import numpy as np

class KNNClassifier:
    def __init__(self, k=5, distance="euclidean", weighted=False):
        if distance not in ["euclidean", "manhattan"]:
            raise ValueError("distance must be 'euclidean' or 'manhattan'")
        
        self.k = k # number of neighbors
        self.distance = distance # distance measure
        self.weighted = weighted # weighted prediction or majority vote

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
            # get distances and numerate them
            distances = [(self._compute_distance(x, x_train), idx)for idx, x_train in enumerate(self.X_train)]

            distances_sorted = sorted(distances, key=lambda t: t[0])
            neighbors = distances_sorted[:self.k]

            neighbor_indices = [idx for (_, idx) in neighbors] # only get indices
            neighbor_distances = np.array([dist for (dist, _) in neighbors]) # only get distances
            neighbor_labels = self.y_train[neighbor_indices] # get classes of neighbors

            if self.weighted:
                weights = 1 / (neighbor_distances + 1e-9)
                class_scores = {}

                for label, w in zip(neighbor_labels, weights):
                    class_scores[label] = class_scores.get(label, 0) + w

                pred = max(class_scores, key=class_scores.get) # weighted prediction bases on distance

            else:
                # majority vote prediction
                values, counts = np.unique(neighbor_labels, return_counts=True)
                pred = values[np.argmax(counts)]

            predictions.append(pred)

        return np.array(predictions)

    def get_neighbors(self, x):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        x = np.array(x)

        distances = [(self._compute_distance(x, x_train), idx)for idx, x_train in enumerate(self.X_train)]

        distances_sorted = sorted(distances, key=lambda t: t[0])
        neighbors = distances_sorted[:self.k]

        neighbor_indices = [idx for (_, idx) in neighbors]
        neighbor_distances = np.array([dist for (dist, _) in neighbors])

        return self.X_train[neighbor_indices], self.y_train[neighbor_indices], neighbor_distances
    
if __name__ == "__main__":
    np.random.seed(1250)

    # Create 2 classes
    n = 100
    X1 = np.random.randn(n, 2) + np.array([2, 2])
    X2 = np.random.randn(n, 2) + np.array([7, 7])

    X = np.vstack((X1, X2))
    y = np.array([0]*n + [1]*n)

    # New point
    new_x = np.array([[5, 4]])

    # Train classifier
    knn = KNNClassifier(k=7, distance="euclidean", weighted=True)
    knn.fit(X, y)

    pred = knn.predict(new_x)[0]

    neighbors_X, neighbors_y, _ = knn.get_neighbors(new_x[0])

    # Plot
    plt.figure(figsize=(6,6))

    # Plot classes 
    plt.scatter(X1[:,0], X1[:,1], color="blue", alpha=0.6, label="Class 0")
    plt.scatter(X2[:,0], X2[:,1], color="red", alpha=0.6, label="Class 1")
    plt.scatter(new_x[0,0], new_x[0,1], color="black", s=100, label=f"Prediction: {pred}")

    # highlight neighbors
    plt.scatter(neighbors_X[:,0], neighbors_X[:,1],edgecolors="black", facecolors="none", s=120, label="Neighbors")

    plt.title("kNN Classification")
    plt.legend()
    plt.show()