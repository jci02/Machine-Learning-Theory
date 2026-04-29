import numpy as np
from matplotlib import pyplot as plt

# Stable sigmoid
def sigmoid(z):
    z = np.clip(z, -500, 500) # numerical stability
    return 1 / (1 + np.exp(-z))


class LogRegGD:
    def __init__(self, lr=0.02, max_iter=1000):
        if not isinstance(lr, (float, int)) or not isinstance(max_iter, int):
            raise ValueError("Parameter types are incompatible")

        self.lr = lr
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.is_fitted = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != len(y):
            raise ValueError("X has incompatible shape with y")

        n, p = X.shape

        # initialize parameters
        self.weights = np.zeros(p)
        self.bias = 0

        # gradient descent
        for _ in range(self.max_iter):
            # compute scores and apply sigmoid
            z = self.bias + X @ self.weights
            pred = sigmoid(z)

            error = pred - y

            # gradients
            grad_w = (1 / n) * (X.T @ error)
            grad_b = (1 / n) * np.sum(error)

            # update parameters
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return sigmoid(self.bias + X @ self.weights) # get raw probability

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int) # get class label predictions with treshold 0.5
    

if __name__ == "__main__":
    np.random.seed(1406)
    n = 100

    # Split samples
    n0 = int(0.6 * n)
    n1 = n - n0

    # Generate features from normal distributions
    X0 = np.random.normal(loc=[2, 2], scale=2.0, size=(n0, 2))  # Class 0
    X1 = np.random.normal(loc=[7, 7], scale=2.0, size=(n1, 2))  # Class 1

    # Combine
    X = np.vstack((X0, X1))
    y = np.array([0]*n0 + [1]*n1)

    # Train model 
    model = LogRegGD()
    model.fit(X,y)
    bias = model.bias
    coeffs = model.weights
    print(f"bias term is {bias}")
    print(f"coefficients are {coeffs}")
    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print(f"Training accuracy = {accuracy:.4f}")

    # Plot data (colored by class) 
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.scatter(X[y==0][:,0], X[y==0][:,1], label="Class 0",color="green")
    plt.scatter(X[y==1][:,0], X[y==1][:,1], label="Class 1",color="red")

    # Decision boundary 
    # b + theta1*x1 + theta2*x2 = 0 <=> x2 = -(theta1/theta2)x1 - b/theta2
    x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)

    if coeffs[1] != 0: # avoid diving by 0
        y_vals = -(coeffs[0]/coeffs[1]) * x_vals - bias/coeffs[1]
        plt.plot(x_vals, y_vals, label="Decision Boundary",color="blue")
    else:
        # vertical line case
        plt.axvline(x = -bias/coeffs[0], label="Decision Boundary")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Logistic Regression Decision Boundary")
    plt.legend()

    plt.subplot(1,2,2)
    z = bias + X @ coeffs 
    sig = sigmoid(z)
    plt.axhline(0.5,linestyle="--")
    plt.axvline(0,linestyle="--")
    plt.scatter(z[y==0],sig[y==0],color="green")
    plt.scatter(z[y==1],sig[y==1],color="red")
    z_line=np.linspace(min(z),max(z),200)
    sig_line=sigmoid(z_line)
    plt.plot(z_line,sig_line,color="black",label="Sigmoid function")
    plt.xlabel(r"$\theta_0 + \theta^\top x$")
    plt.ylabel(r"$sigmoid(\theta_0 + \theta^\top x)$")
    plt.title("Logistic Regression Sigmoid")
    plt.legend()


    plt.show()