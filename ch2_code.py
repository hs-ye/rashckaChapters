# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 07:49:25 2017
Raschka - Python Machines Learning
Follow along

Chapter 2
@author: yehan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand
from matplotlib.colors import ListedColormap

# %% Perceptrons
class Perceptron(object):
	"""Basic Perceptron Classifier
	Parameters
	------------
	eta : float
	Learning rate (between 0.0 and 1.0)
	n_iter : int
	Passes over the training dataset.

	Attributes
	-----------
	w_ : 1d-array
	Weights after fitting.
	errors_ : list
	Number of misclassifications in every epoch. No self stopping objective fn,
	have to rely on looking at the minimum misclassification to determine optimal
	iterations. This is a v. basic algorithm

	Note about the (object) in class header: can omit in py 3.x, only
	eeded for py 2.x to declare base inheritance from the object class
	"""

	def __init__(self, eta=0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		self.w_ = np.zeros(1 + X.shape[1])  # initial weights matrix
		self.errors_ = []

		for _ in range(self.n_iter):  # just iterates, don't do anything with _
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update  # 0 is intercept term, no xi needed
#				print(self.w_)
#				print(update != 0.0)
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		""" Calculate Net Input """
		return np.dot(X, self.w_[1:]) + self.w_[0]  # total predicted output

	def predict(self, X):
		""" return class label after unit step """
		return np.where(self.net_input(X) >= 0.0, 1, -1)


# %%

df_iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df_iris.head()
df_iris.tail()
y = df_iris.iloc[0:100, 4].values  # get just the labels, no index
# y = np.where(y == 'Iris-setosa', 1, 0)  # this casting won't work if erros are 0
y = np.where(y == 'Iris-setosa', -1, 1)  # cast to -1 1 encoding, needed for errors = 0
X = df_iris.iloc[0:100, [0,2]].values  # get just 2 cols of predictors, 2D model only
# %%
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='vers')
plt.scatter(X[50:, 0], X[50:, 1], color="blue", marker='x', label='sets')
plt.xlabel('petal len')
plt.ylabel('sepal len')
plt.legend(loc="lower right")
# plt.show()  # don't need this for spyder/ipython, which autoplots

# %%
mod_pcp = Perceptron()
mod_pcp.fit(X, y)
plt.plot(range(1, len(mod_pcp.errors_) + 1), mod_pcp.errors_)

# %% make a chart by hand, plotting all the colours

def plot_decision_regions(X, y, classifier, resolution=0.02):
	'''Testing:
	classifier = Perceptron()
	classifier.fit(X, y)
	resolution=0.02
	'''

	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')  # pre-define types of markers and colours to plot
	colours = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colours[:len(np.unique(y))])  # make as many colours as classes

	# plot decision surface, along 2 axis
	marg = 1  # margin, to add to graph at either end of scales
	x1_min, x1_max = X[:, 0].min() - marg, X[:, 0].max() + marg  # the min & max of first column
	x2_min, x2_max = X[:, 1].min() - marg, X[:, 1].max() + marg  # the min & max of 2nd column
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
							np.arange(x2_min, x2_max, resolution)) # arange - similar to linspace
	#meshgrid: Creates
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # classifier is passed in
	# Need the classifier to predict output value at each co-ordinate of
	Z = Z.reshape(xx1.shape)  # turn this into a 2d array of prediction outputs
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)  # makes the contour map based on colours
	plt.xlim(xx1.min(), xx1.max())  # axis
	plt.ylim(xx2.min(), xx2.max())

	# plot class samples
	for idx, cl in enumerate(np.unique(y)):  # gets the unique outcome classes
		print(idx, cl)  # their value and what order they're in
		# used to determine how to plot them
		plt.scatter(x=X[y == cl, 0], y=X[y ==cl, 1], alpha=0.8,
					c=cmap(idx), marker=markers[idx], label=cl)

# %%

plot_decision_regions(X, y, classifier = mod_pcp)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


# %% Adeline Gradient Descent

class AdalineGD():
	"""ADAptive LInear NEuron classifier.
	Parameters
	------------
	eta : float
	Learning rate (between 0.0 and 1.0)
	n_iter : int
	Passes over the training dataset.

	Attributes
	-----------
	w_ : 1d-array
	Weights after fitting.
	errors_ : list
	Number of misclassifications in every epoch.
	"""
	def __init__(self, eta=0.01, n_iter=50):
	self.eta = eta
	self.n_iter = n_iter

	def fit(self, X, y):
	""" Fit training data.

	Parameters
	----------
	X : {array-like}, shape = [n_samples, n_features]
	Training vectors,
	where n_samples is the number of samples and
	n_features is the number of features.
	y : array-like, shape = [n_samples]
	Target values.

	Returns
	--------
	self : object
	"""
	self.w_ = np.zeros(1 + X.shape[1])  # weights, ncols = n features
	self.cost_ = []

	for i in range(self.n_iter):
		output = self.net_input(X)
		errors = (y - output)  # n x 1 column vector
		self.w_[1:] += self.eta * X.T.dot(errors)  # X transform (k x n) dot multiply errors matrix
		# above: this is the gradient update, see book for derivation
	    self.w_[0] += self.eta * errors.sum()
		cost = (errors**2).sum() / 2.0  # errors squared, summed, divided by 2
		self.cost_.append(cost)  # append cost per iteration
	return self

	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		"""Compute linear activation, just a straight return of the linear net input"""
		return self.net_input(X)

	def predict(self, X):
		"""Return class label after unit step"""
		return np.where(self.activation(X) >= 0.0, 1, -1)


# %% Make random data - extra exercise, not in book
# different ways the data frames work....
rnd_col = rand.sample(range(1,100),10)
rnd_col2 = rand.sample(range(1,100),10)
X = pd.DataFrame(rand.sample(range(1,100),10), rand.sample(range(1,100),10))  # 1st is treated as index
X = pd.DataFrame([rand.sample(range(1,100),10), rand.sample(range(1,100),10)]) # 2 rows instead of cols
X = pd.DataFrame(list(zip(rand.sample(range(1,100),10), rand.sample(range(1,100),10)))) # normal one, but verbose
X = pd.concat([rnd_col, rnd_col], axis=1, keys = ['col1', 'col2'])  # won't work, only for numpy arrays
#
rnd_dat = [('col1', rnd_col), ('col2', rnd_col2)]  # set up a proper column based list
X = pd.DataFrame.from_items(rnd_dat)  # and generate raw data from it
y = pd.DataFrame(rand.choices([-1, 1], k=10))  # only in python 3.6...
y = pd.DataFrame([rand.choice([-1, 1]) for _ in range(10)])  # list comprehend in p3.5 and below

# %Non DF versions of X/Y
X = list(zip(rand.sample(range(1,100),10), rand.sample(range(1,100),10)))  # native list of tuples, works in the zip below
X = np.array([rand.sample(range(1,100),10), rand.sample(range(1,100),10)]) # doesn't work...
X = np.array([rand.sample(range(1,100),10), rand.sample(range(1,100),10)]).T # now works, wrong direction
y = [rand.choice([-1, 1]) for _ in range(10)]

# Combine X and y as per the raschka estimator .fit() methods
list(zip(X, y))   # makes each row a tuple of all the X vars accompanied by the y output


# %% Making standardised data
X_std = np.copy(X)  # copy of X
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()  # manually normalise X
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# %% Adeline - Stochastic Gradient Descent
from numpy.random import seed

class AdalineSGD(object):
	"""ADAptive LInear NEuron classifier.

	Parameters
	------------
	eta : float
	Learning rate (between 0.0 and 1.0)
	n_iter : int
	Passes over the training dataset.

	Attributes
	-----------
	w_ : 1d-array
	Weights after fitting.
	errors_ : list
	Number of misclassifications in every epoch.
	shuffle : bool (default: True)
	Shuffles training data every epoch
	if True to prevent cycles.
	random_state : int (default: None)
	Set random state for shuffling
	and initializing the weights.

	"""
	def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle
		if random_state:
			seed(random_state)

	def fit(self, X, y):
		""" Fit training data.
		Parameters
		----------
		X : {array-like}, shape = [n_samples, n_features]
		Training vectors, where n_samples
		is the number of samples and
		n_features is the number of features.
		y : array-like, shape = [n_samples]
		Target values.

		Returns
		-------
		self : object
		"""
		self._initialize_weights(X.shape[1])
		self.cost_ = []
		for i in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)
			cost = []
			for xi, target in zip(X, y):
				cost.append(self._update_weights(xi, target))
			avg_cost = sum(cost)/len(y)
			self.cost_.append(avg_cost)
		return self

	def partial_fit(self, X, y):
		"""Fit training data without reinitializing the weights"""
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi, target in zip(X, y):
				self._update_weights(xi, target)
		else:
			self._update_weights(X, y)
		return self

	def _shuffle(self, X, y):
		"""Shuffle training data"""
		r = np.random.permutation(len(y))
		return X[r], y[r]

	def _initialize_weights(self, m):
		"""Initialize weights to zeros"""
		self.w_ = np.zeros(1 + m)
		self.w_initialized = True

	def _update_weights(self, xi, target):
		"""Apply Adaline learning rule to update the weights"""
		output = self.net_input(xi)
		error = (target - output)
		self.w_[1:] += self.eta * xi.dot(error)
		self.w_[0] += self.eta * error
		cost = 0.5 * error**2
		return cost

	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		"""Compute linear activation"""
		return self.net_input(X)

	def predict(self, X):
		"""Return class label after unit step"""
		return np.where(self.activation(X) >= 0.0, 1, -1)

# %%

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()




































