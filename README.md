## Python Library for Logistic Regression

* Logistic Regression is the Supervised Learning Algorithm for solving Classification problems like categorizing email as spam or not spam.

* Install the library using the command pip3 install git+https://github.com/rgeetha2010/LogisticRegression.git
  
* An example application is provided for reference. 

* Regularization parameter is a control on the fitting parameters. Higher order polynomial expressions give overfitting problems. We add regularization term to penalize parameters of higher order polynomials to get a better fit.

* model = train(X, y, reg_param, num_labels) 
  - X: features of training set
  - y: corresponding labels of training set
  - reg_param: Regularization parameter
  - num_labels: Number of labels/categories/classes
  
* accuracy = predict(model, X, y)
  - model: trained model
  - X: features of training set
  - y: corresponding labels of training set
