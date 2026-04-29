import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from RandomForestRegression import RandomForest

np.random.seed(1510)

# nonlinear data
X = np.sort(np.random.uniform(0, 10, 200)).reshape(-1,1)
y = 3*np.sin(X[:,0]) + 0.5*X[:,0] + np.random.normal(0, 0.5, len(X))

# our forest
my_forest = RandomForest(n_trees=20, min_samples_split=3, max_depth=6)
my_forest.fit(X, y)

# sklearn forest
sk_forest = SkRandomForestRegressor(n_estimators=20, min_samples_split=3, max_depth=6, random_state=1510)
sk_forest.fit(X, y)

# smooth grid
X_grid = np.linspace(X.min(), X.max(), 500).reshape(-1,1)

pred_my = my_forest.predict(X_grid)
pred_sk = sk_forest.predict(X_grid)

# plot
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.scatter(X[:,0], y, alpha=0.5, color="gray")
plt.plot(X_grid[:,0], pred_my, color="red", linewidth=2)
plt.title("Our Random Forest")

plt.subplot(1,2,2)
plt.scatter(X[:,0], y, alpha=0.5, color="gray")
plt.plot(X_grid[:,0], pred_sk, color="blue", linewidth=2)
plt.title("sklearn Random Forest")

plt.tight_layout()
plt.show()