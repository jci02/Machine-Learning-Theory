import numpy as np
import matplotlib.pyplot as plt


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.vars = {}
        self.is_fitted = False


    # Fit
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != len(y):
            raise ValueError("Dimension mismatch between X and y")

        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for c in self.classes:
            X_c = X[y == c]

            self.class_priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-9

        self.is_fitted = True
        return self


    # Gaussian density
    def _gaussian_pdf(self, x, mean, var):
        const = 1 / np.sqrt(2 * np.pi * var)
        exp_term = np.exp(-((x - mean) ** 2) / (2 * var))
        return const * exp_term


    # predict one point
    def _predict_single(self, x):
        posteriors = []

        for c in self.classes:
            log_prior = np.log(self.class_priors[c])

            pdf_vals = self._gaussian_pdf(x, self.means[c], self.vars[c])

            log_likelihood = np.sum(np.log(pdf_vals))

            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]


    # predict all
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return np.array([self._predict_single(x) for x in X])
    

if __name__ == "__main__":

    # Generate toy Gaussian data
    np.random.seed(1723)

    X1 = np.random.multivariate_normal([10, 2], [[3, 1], [1, 2]], 100)
    X2 = np.random.multivariate_normal([20, 7], [[3, 1], [1, 2]], 100)

    X = np.vstack([X1, X2])
    y = np.array([0]*100 + [1]*100)

    # Fit Naive Bayes model
    model_nb = NaiveBayes()
    model_nb.fit(X, y)

    preds_nb = model_nb.predict(X)

    print("Predictions:")
    print(preds_nb)

    print("Naive Bayes Accuracy:", np.mean(preds_nb == y))


    # Create decision grid
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-2, X[:, 0].max()+2, 200),np.linspace(X[:, 1].min()-2, X[:, 1].max()+2, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z_nb = model_nb.predict(grid)
    Z_nb = Z_nb.reshape(xx.shape)


    # Plot
    plt.figure(figsize=(7, 6))

    # filled regions
    plt.contourf(xx, yy, Z_nb, alpha=0.2)

    # boundary
    plt.contour(xx, yy, Z_nb, levels=[0.5], linestyles="--", colors="black")

    # data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", color="green")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", color="red")

    plt.title("Gaussian Naive Bayes Decision Boundary")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.show()