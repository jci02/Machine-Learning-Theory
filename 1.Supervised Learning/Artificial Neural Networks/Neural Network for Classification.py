import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork:

    def __init__(self, input_size=2, hidden_size=2, output_size=3, seed=1705):
        np.random.seed(seed)
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros(output_size)

    # Activation functions
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    # Forward pass
    def forward(self, x, t=None):
        a1 = self.W1 @ x + self.b1
        z1 = np.maximum(0, a1)
        a2 = self.W2 @ z1 + self.b2
        y = self.softmax(a2)
        L = None

        if t is not None:
            L = -np.sum(t * np.log(y + 1e-12))

        return a1, z1, a2, y, L


    # Backpropagation
    def backward(self, x, t, lr=0.01):
        a1, z1, a2, y, L = self.forward(x, t)

        # Output layer
        delta2 = y - t
        dLdb2 = delta2
        dLdW2 = np.outer(delta2, z1)

        # Hidden layer
        relu_grad = (a1 > 0).astype(float)

        delta1 = (self.W2.T @ delta2) * relu_grad

        dLdb1 = delta1
        dLdW1 = np.outer(delta1, x)

        # Gradient descent update
        self.W1 -= lr * dLdW1
        self.b1 -= lr * dLdb1
        self.W2 -= lr * dLdW2
        self.b2 -= lr * dLdb2

        return L


    # Training
    def fit(self, X, T, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            total_loss = 0

            for x, t in zip(X, T):
                total_loss += self.backward(x, t, lr)

            avg_loss = total_loss / len(X)

            if epoch % 100 == 0:
                print(f"epoch={epoch:4d}, loss={avg_loss:.6f}")


    # Prediction
    def predict(self, x):
        _, _, _, y, _ = self.forward(x)
        return np.argmax(y)


    # Probabilities
    def predict_proba(self, x):
        _, _, _, y, _ = self.forward(x)
        return y


    # Accuracy
    def score(self, X, y):
        correct = 0

        for x, label in zip(X, y):
            if self.predict(x) == label:
                correct += 1

        return correct / len(X)
    

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()

    # Sepal Width and Petal Width
    X = iris.data[:, [1, 3]]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Labels
    y = iris.target

    # One-hot encoding
    T = np.eye(3)[y]

    # Train/test split
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X,T,y,test_size=0.2,random_state=1705)

    # Create neural network
    nn = NeuralNetwork()

    # Train
    nn.fit(X_train, T_train, epochs=1000, lr=0.01)

    # Accuracy
    accuracy = nn.score(X_test, y_test)

    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Example prediction
    pred = nn.predict(X_test[0])

    print("\nPrediction:", iris.target_names[pred])
    print("Actual:    ", iris.target_names[y_test[0]])

    # Class probabilities
    probs = nn.predict_proba(X_test[0])

    print("\nProbabilities:", probs)
    print("Target:", T_test[0])