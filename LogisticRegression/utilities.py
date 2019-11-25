import pandas as pd
import numpy as np
from scipy import optimize
import scipy.io


# Sigmoid Function
def sigmoid(z):
	g = 1.0 / (1.0 + np.exp(-z))
	return g

# Hypothesis Function
def hypothesis(X, theta):
	[m, n] = np.shape(X)
	theta = theta.reshape(1, n)
	z = X @ (theta.T) # Matrix multiplication
	h = sigmoid(z)
	return h

# Cost Function
def costFunction(theta, X, y, lmbda):
	J = 0
	[m, n] = np.shape(X)
	theta = theta.reshape(1, n)
	first_term = -y.T.dot(np.log(hypothesis(X, theta)))
	second_term = (1 - y).T.dot(np.log(1 - hypothesis(X, theta)))
	sum_term = first_term - second_term
	avg_cost = sum_term / m
	reg_term = (lmbda / (2 * m)) * (np.sum(theta.dot(theta.T)))
	J = avg_cost + reg_term
	return J

# Gradient Descent
def gradientDescent(theta, X, y, lmbda):
	[m, n] = np.shape(X)
	theta = theta.reshape(1, n)
	grad = np.zeros((1, n))
	grad1 = (hypothesis(X, theta) - y).T.dot(X[:,0:1])/m
	grad[:,0:1] = grad1
	for i in range(1, n):
		grad[:,i:] = ((hypothesis(X, theta) - y).T.dot(X[:,i:i+1])/m) + ((lmbda / m) * theta[:,i:i+1])
	grad = grad.T
	grad = grad.ravel()
	return grad

# Classification of data
def oneVsAllClassification(X, y, theta, lmbda, num_labels):
	[m, n] = np.shape(X)
	theta = theta.reshape(1, n)
	result = optimize.fmin_cg(costFunction, fprime = gradientDescent, x0 = theta, args = (X, y, lmbda), maxiter = 50, disp = False, full_output = True)
	return result

# Predict values
def predict(all_theta, X, y):
	m = np.shape(X)[0]
	X = np.insert(X,0,1,axis=1)
	num_labels = np.shape(all_theta)[0]
	p = np.zeros((m, 1))
	h_max = sigmoid(X.dot(all_theta.T))
	for i in range(1, m):
		max_prob = np.argmax(h_max[i-1, :])
		p[i-1] = max_prob
	# Accuracy
	num_correct = 0

	for i in range(m):
		if(p[i] == y[i]):
			num_correct = num_correct + 1
	accuracy = (num_correct / m) * 100
	return accuracy

# Train model
def train(X, y, lmbda, num_labels):
	print("             --------------------------------------------------")
	print("             |                                                |")
	print("             |               Logistic  Regression             |")
	print("             |                                                |")
	print("             --------------------------------------------------")
	print()
	[m,n] = np.shape(X)
	X = np.insert(X,0,1,axis=1)
	initial_theta = np.zeros((1, n+1)) 
	all_theta = np.zeros((num_labels, n+1))
	for i in range(num_labels):
		print("Training the Model for label" ,i ,"...")
		print("")
		y_bool = np.matrix([1 if label == i else 0 for label in y])
		y_bool = y_bool.T
		updated_theta = oneVsAllClassification(X, y_bool, initial_theta, lmbda, num_labels)
		all_theta[i,:] = updated_theta[0]
	return all_theta
  
