import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px # for plotting
import plotly.graph_objects as go # for plotting
import plotly.io as pio

class AdaBoostClassificationTree:
    """
    AdaBoost for binary classification using DecisionTreeClassifier stumps.
    Labels must be {-1, +1}
    """

    def __init__(self, n_learners=50, max_depth=1):
        self.n_learners = n_learners
        self.max_depth = max_depth

        self.learners = []
        self.betas = []

    def fit(self, X, y):
        """
        Train AdaBoost
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Labels must be {-1, +1}
        """

        n_samples = X.shape[0]

        # Step 1: initialize equal weights
        weights = np.ones(n_samples) / n_samples

        for m in range(self.n_learners):

            # Step 2: fit weak learner with sample weights
            stump = DecisionTreeClassifier(max_depth=self.max_depth,random_state=1020)

            stump.fit(X, y, sample_weight=weights) # sample_weight=weights to compute split quality with weighted Gini 

            # Predictions
            y_pred = stump.predict(X)

            # Step 3: weighted classification error
            missclassified = (y != y_pred)

            error = np.sum(weights * missclassified)

            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Step 4: learner weight
            beta = 0.5 * np.log((1 - error) / error)

            # Step 5: update sample weights
            weights *= np.exp(-beta * y * y_pred)

            # Step 6: normalize weights
            weights /= np.sum(weights)

            # Store learner
            self.learners.append(stump)
            self.betas.append(beta)

            print(f"Iteration {m+1}")
            print(f"Weighted error: {error:.4f}")
            print(f"Beta: {beta:.4f}")
            print("-" * 40)

    def predict_scores(self, X):
        """
        Compute f(x) = sum(beta_m * h_m(x))
        """

        scores = np.zeros(X.shape[0])

        for beta, learner in zip(self.betas, self.learners):
            scores += beta * learner.predict(X)

        return scores

    def predict(self, X):
        """
        Final prediction: sign(f(x))
        """

        scores = self.predict_scores(X)

        return np.sign(scores)


if __name__ == "__main__":
    # Example Usage
    # Create binary classification dataset
    X, y = make_classification(n_samples=500,n_features=10,n_informative=5,n_redundant=0,random_state=1020)

    # Convert labels from {0,1} -> {-1,+1}
    y = np.where(y == 0, -1, 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1020)

    # Train AdaBoost
    model = AdaBoostClassificationTree(n_learners=50,max_depth=1)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}")

    # Plot
    pio.renderers.default = "browser"

    # Predictions on ALL data
    y_all_pred = model.predict(X)

    # Convert labels for nicer plotting
    y_true_plot = np.where(y == -1, 0, 1)
    y_pred_plot = np.where(y_all_pred == -1, 0, 1)

    # Correct / Incorrect
    correctness = np.where(y == y_all_pred,"Correct","Incorrect")


    # Plot true labels
    fig_true = px.scatter(x=X[:, 0],y=X[:, 1],color=y_true_plot.astype(str),title="True Labels",
                          labels={"x": "Feature 1","y": "Feature 2","color": "Class"})

    fig_true.update_traces(marker=dict(size=8))
    fig_true.show()

    # Plot predicted labels
    fig_pred = px.scatter(x=X[:, 0],y=X[:, 1],color=y_pred_plot.astype(str),title="Predicted Labels (AdaBoost)",
                          labels={"x": "Feature 1","y": "Feature 2","color": "Predicted Class"})

    fig_pred.update_traces(marker=dict(size=8))
    fig_pred.show()


    # Plot Correct vs Incorrect
    fig_correct = px.scatter(x=X[:, 0],y=X[:, 1],color=correctness,symbol=correctness,title="Correct vs Incorrect Predictions",
                             labels={"x": "Feature 1","y": "Feature 2"})

    fig_correct.update_traces(marker=dict(size=9))
    fig_correct.show()

    # Decision Boundary (using first 2 features)
    h = 0.05 # step size for grids

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    # Create grid with full feature dimension
    grid = np.zeros((xx.ravel().shape[0], X.shape[1]))

    # Only vary first 2 features
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()

    # Predict on mesh
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Contour plot
    fig_boundary = go.Figure()

    fig_boundary.add_trace(go.Contour(x=np.arange(x_min, x_max, h),y=np.arange(y_min, y_max, h),z=Z,opacity=0.4,showscale=False))

    # Scatter points
    fig_boundary.add_trace(go.Scatter(x=X[:, 0],y=X[:, 1],mode='markers',marker=dict(color=y_true_plot,size=7),
                                      text=[f"True: {t}, Pred: {p}"for t, p in zip(y_true_plot, y_pred_plot)]))

    fig_boundary.update_layout(title="AdaBoost Decision Boundary",xaxis_title="Feature 1",yaxis_title="Feature 2")

    fig_boundary.show()      