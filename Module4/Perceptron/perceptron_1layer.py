# Perceptron

# Import libraries
from matplotlib import pyplot as plt
import numpy as np
import math
# Dataset
dataset = np.array([[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]])

global x, y, z 
x, y, z = dataset[:, 0], dataset[:, 1], dataset[:, 2]

def plot(num, prediction, error):
	# plt.subplot(3, 2, num)
	plt.scatter(x[prediction == 0.0], y[prediction == 0.0], label = "0", c = "red", )
	plt.scatter(x[prediction == 1.0], y[prediction == 1.0], label = "1", c = "blue")
	plt.legend(loc='lower left')
	plt.title(f'Epoch {num}')

# Step-function (a threshold)
def step_function(num):
	return 1.0 if num >= 0.0 else 0.0

# Relu activation function
def relu(num):
	return max(0, num)

# Sigmoid activation function
def sigmoid(num):
	return  1/(1+math.e**(-num))

# Linear activation function
def identity(num):
	return num

# Make a prediction with weights
def predict(row, weights):
	result = weights[0] # add the bias term
	for i in range(len(row)-1):
		result += weights[i + 1] * row[i] # calculates a weight sum
	return step_function(result)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.001 for i in range(len(train[0]))]
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
		# plot(epoch + 1, np.array([predict(row, weights) for row in train]), sum_error)
		print(f'>epoch={epoch + 1}, lrate={round(l_rate, 3)}, error={round(sum_error, 3)}')
		errors.append(sum_error)
	return weights, errors

# Calculate weights
l_rate = 0.01
n_epoch = 100
weights, errors = train_weights(dataset, l_rate, n_epoch)
print(weights)


plot(1, z, 0)
plt.show()
plt.plot(errors)
plt.show()