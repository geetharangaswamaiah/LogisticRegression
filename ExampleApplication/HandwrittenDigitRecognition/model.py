import pandas as pd
import scipy.io
from logistic_regression.utilities import train
from logistic_regression.utilities import predict

# Read data
datafile = 'example_data.mat' # Subset of MNIST handwritten digit images
mat = scipy.io.loadmat(datafile) 
X = mat['X']
y = mat['y']

# Set regularization parameter
reg_param = 0.1

# number of labels (classes)
num_labels = 10 # Example Data is a subset of MNIST handwritten digit images, so 10 classes (0 to 9)

model = train(X, y, reg_param, num_labels)

print("Accuracy -->" , predict(model, X, y))


