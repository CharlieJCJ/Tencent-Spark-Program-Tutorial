import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import plotly.graph_objects as go


df = pd.read_csv('./weatherww2.csv')
df = df[['MinTemp', 'MaxTemp']]
# display(df.head(10))
df.plot.scatter('MinTemp', 'MaxTemp', title = "weather dataset");

# define the vectorized MSE cost function 
def mse_cost(predictions, target):
    N = predictions.shape[0]
    diff = predictions.ravel() - target.ravel()
    cost = np.dot(diff, diff.T) / N
    return cost

# define the prediction for a simple linear model
def LinearModel(thetas, X):
    # normalize add bias term
    X = (X - X.mean()) / X.std()
    X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
    return np.dot(X, thetas)

# initialize data structures
vis_df = df.sample(frac=0.1)
y = vis_df.iloc[:, 1].to_numpy().reshape(-1, 1)
X = vis_df.iloc[:, 0].to_numpy().reshape(-1, 1)

# grid search over "all" possible theta values and compute cost
start, end, step = -200, 200, 5
thetas_0, thetas_1 = np.arange(start, end, step), np.arange(start, end, step)

# loop over the all the parameter pairs and create a list of all possible pairs
thetas_lst = []
for theta_0 in thetas_0:
    for theta_1 in thetas_1:
        thetas_lst.append(np.array([theta_0, theta_1]).reshape(-1, 1))

linear_cost_lst = []       
for thetas in thetas_lst:
    # get prediction from our model
    pred_linear = LinearModel(thetas, X)
    # keep track of the cast per parameter pairs
    linear_cost_lst.append(mse_cost(pred_linear, y))

# arrange the costs back to a square matrix grid
axis_length = len(np.arange(start, end, step))
linear_cost_matrix = np.array(linear_cost_lst).reshape(axis_length, axis_length)


# plot the surface plot with plotly's Surface
fig = go.Figure(data=go.Surface(z=linear_cost_matrix,
                                x=thetas_0,
                                y=thetas_1))

# add a countour plot
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

# annotate the plot
fig.update_layout(title='Linear Model MSE Cost Surface',
                  scene=dict(
                    xaxis_title='theta_0 (intercept)',
                    yaxis_title='theta_1 (slope)',
                    zaxis_title='MSE Cost'),
                  width=700, height=700)


# 这里会打开一个新的网页，有interactive交互
fig.show()
plt.show()
