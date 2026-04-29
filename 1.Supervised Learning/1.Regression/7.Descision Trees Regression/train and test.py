import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor as SkDecisionTreeRegressor

from DecisionTreeRegressor import DecisionTreeRegressor


np.random.seed(1507)

# nonlinear regression data
X = np.sort(np.random.uniform(0, 10, 150)).reshape(-1,1)
y = 3*np.sin(X[:,0]) + 0.5*X[:,0] + np.random.normal(0, 0.4, len(X))

# our model
my_tree = DecisionTreeRegressor(min_samples_split=4, max_depth=4)
my_tree.fit(X, y)
preds_my = my_tree.predict(X)

# sklearn model
sk_tree = SkDecisionTreeRegressor(min_samples_split=4, max_depth=4, random_state=1507)
sk_tree.fit(X, y)
preds_sk = sk_tree.predict(X)

# smooth grid
X_grid = np.linspace(X.min(), X.max(), 400).reshape(-1,1)
grid_preds_my = my_tree.predict(X_grid)
grid_preds_sk = sk_tree.predict(X_grid)

# plotting
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.scatter(X[:,0], y, color="gray", alpha=0.6, label="Data")
plt.plot(X_grid[:,0], grid_preds_my, color="red", linewidth=2, label="Our Tree")
plt.title("Our Decision Tree Regressor")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.subplot(1,2,2)
plt.scatter(X[:,0], y, color="gray", alpha=0.6, label="Data")
plt.plot(X_grid[:,0], grid_preds_sk, color="blue", linewidth=2, label="sklearn Tree")
plt.title("sklearn Decision Tree Regressor")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()