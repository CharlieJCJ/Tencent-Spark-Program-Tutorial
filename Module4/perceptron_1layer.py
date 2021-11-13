# Perceptron

# Import libraries
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn import datasets

# Dataset
train, label = datasets.make_blobs(n_samples=500,n_features=2,
                           centers=2,cluster_std=1.5,
                           random_state=2)

global x, y, z 
x, y, z = train[:, 0], train[:, 1], label
dataset = [[x[i], y[i], z[i]] for i in range(len(x))]
l_rate = 0.02
n_epoch = 10

# Helper plot function
def plot(num, prediction):
	plt.subplot(1, 2, 1)
	plt.scatter(x[z <0.5], y[z <0.5], label = "0", c = "red", )
	plt.scatter(x[z >=0.5], y[z >=0.5], label = "1", c = "blue")
	plt.legend(loc='lower left')
	plt.title("True Dataset")


	plt.subplot(1, 2, 2)
	plt.scatter(x[prediction < 0.5], y[prediction < 0.5], label = "0", c = "red", )
	plt.scatter(x[prediction >= 0.5], y[prediction >= 0.5], label = "1", c = "blue")
	plt.legend(loc='lower left')
	plt.title(f'Epoch {num}')
	plt.show()


# *** Activation functions: *** #
# Step-function (a threshold)
def step_function(num):
	return 1.0 if num >= 0.0 else 0.0

# Relu activation function
def relu(num):
	return max(0, num)

# Sigmoid activation function
def sigmoid(num):
	return  1/(1+math.e**(-num))

# Linear function
def identity(num):
	return num

# ****************************** # 

# Make a prediction with weights
def predict(row, weights):
	result = weights[0] # add the bias term
	for i in range(len(row)-1):
		result += weights[i + 1] * row[i] # calculates a weight sum
	return sigmoid(result)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [-10 for i in range(len(train[0]))]
	errors = []
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[2] - prediction
			sum_error += error ** 2
			weights[0] = weights[0] + l_rate * error   # Bias term
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print(f'>epoch={epoch + 1}, lrate={round(l_rate, 3)}, error={round(sum_error, 3)}')
		predictions = np.array([predict(row, weights) for row in train])
		plot(epoch, predictions)
		errors.append(sum_error)
	return weights, errors

# Calculate weights
weights, errors = train_weights(dataset, l_rate, n_epoch)
print(weights)


plt.plot(errors)
plt.title("Cost function")
plt.show()