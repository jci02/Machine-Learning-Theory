import numpy as np # random numbers and arrays
import plotly.express as px # for plotting
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # for Linear Regression
from sklearn.model_selection import train_test_split # to split dataset into train and test set
from sklearn.datasets import make_regression # create Regression toy data
import plotly.io as pio # set where plots are shown
from LinearRegression import LinReg, mse
from LinearRegressionGD import LinRegGD
import plotly.graph_objects as go # for plotting
import plotly.colors as pc # for more color variety using plotly.express as px

x, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=1925)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2025)

plt.figure(figsize=(10,10))
cmap = plt.get_cmap('viridis')
plt.subplot(1,2,1)
plt.scatter(x=x_train.flatten(), y=y_train,color=cmap(0.9),label="Train Data points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Train data points")
plt.legend()

model = LinReg(copy_X=False)
model.fit(x_train,y_train)
print(f"Our solution:bias = {model.bias:.2f} and coefficients={model.weights[0]:.4f}")

model2 = LinearRegression(copy_X=False,fit_intercept=True)
model2.fit(x_train,y_train)
print(f"Pythons solution:bias = {model2.intercept_:.2f} and coefficients={model2.coef_[0]:.4f}")

y_hat = model.predict(x_test)
print(f"MSE={mse(y_test,y_hat)}")
plt.subplot(1,2,2)
plt.scatter(x=x_test.flatten(), y=y_test,color=cmap(0.9),label="Test Data points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Test data points")
plt.plot(x_test,y_hat,color="red",label="Fitted line")
plt.legend()



X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=1822)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1713)
pio.renderers.default = "browser" # to show plot in browser 
fig = px.scatter_3d(x=X_train[:,0],y=X_train[:,1],z=y_train,labels={"x":"x1","y":"x2","z":"x3"})

model = LinReg(intercept=False)
model.fit(X_train,y_train)
print(f"bias = {model.bias:.2f} and coefficients={np.round(model.weights,3)}")

model2 = LinearRegression(fit_intercept=False)
model2.fit(X_train,y_train)
print(f"Pythons:bias = {model2.intercept_:.2f} and coefficients={np.round(model2.coef_,3)}")

# Create grid for surface
x1_range = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 20)
x2_range = np.linspace(X_train[:,1].min(), X_train[:,1].max(), 20)

x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Flatten grid for prediction
grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# Predict using your model
y_grid = model.predict(grid).reshape(x1_grid.shape)

# Add surface to figure
fig.add_trace(go.Surface(x=x1_grid,y=x2_grid,z=y_grid,opacity=0.6,showscale=False,showlegend=True,name="Fitted plane"))
fig.show() 



X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=925)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1050)
model = LinRegGD(intercept=True)
model.fit(X,y)
print(f"Estimated bias={model.bias} and weights={model.weights}:")

# Plot data points
fig = px.scatter_3d(x=X[:,0], y=X[:,1], z=y,title="Data",labels={'x':'x1','y':'x2','z':'y'})

colors = pc.qualitative.Plotly  # nice default palette
for idx, i in enumerate([0, 10, 20, 50, 100, len(model.theta_history)-1]):
    
    # Grid
    grid_x1, grid_x2 = np.meshgrid(np.linspace(X[:,0].min(), X[:,2].max(), 30),np.linspace(X[:,2].min(), X[:,2].max(), 30))

    grid_X = np.column_stack((np.ones(grid_x1.size),grid_x1.ravel(),grid_x2.ravel()))

    # Predictions
    surface_pred = (grid_X @ model.theta_history[i]).reshape(grid_x1.shape)

    # Pick color
    color = colors[idx % len(colors)]

    # Create constant colorscale
    colorscale = [[0, color], [1, color]]

    fig.add_trace(go.Surface(x=grid_x1,y=grid_x2,z=surface_pred,opacity=0.5,
                             showscale=False,colorscale=colorscale,name=f"Iteration {i}",showlegend=True))


fig.update_layout(width=700, height=500)
fig.show()


plt.show() # plt.show() is a blocking GUI call in normal Python scripts: execution pauses there until the plot window is closed
plt.tight_layout()