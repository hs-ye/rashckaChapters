# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 07:33:40 2017

Raschka - Python Machines Learning
Follow along

Chapter 3

@author: yehan
"""

from sklearn import datasets
import numpy as np

# %% Load/inspect data

iris = datasets.load_iris()  # didn't know you could get this for free...
X = iris.data[:, [2, 3]]
y = iris.target  # full list of outputs
np.unique(y)  # target already stored as int


# %% Make training split and scale
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=0)  # test = 0.3, train = 0.7

sc = StandardScaler()  # normalises data
sc.fit(X_train)  # get mean µ and std dev σ of training data
X_train_std = sc.transform(X_train)  # apply to both training and test data
X_test_std = sc.transform(X_test)

# %% Fit perceptron and predict
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0, n_jobs=2)  # built in multi-core?
ppn.fit(X_train_std, y_train)

y_prd = ppn.predict(X_test_std)
print('Prediction accuracy: {0}'.format((y_test == y_prd).sum() / len(y_test)))

# %% Pre-made metrics utilities
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import roc_curve
acc(y_test, y_prd)
auc(y_test,y_prd)
roc_curve(y_test, y_prd)

# %% Make a chart V2, now with test and train separation
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
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
	# highlight test samples
	if test_idx:
		X_test = X[test_idx, :]
		plt.scatter(X_test[:, 0], X_test[:, 1], facecolors='', edgecolors='black',
				    alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

# %% plotting
X_all_std = np.vstack((X_train_std, X_test_std))
y_all = np.hstack((y_train, y_test))
plot_decision_regions(X=X_all_std, y=y_all,
					  classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='lower right')

# %% logistic/sigmoid function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')  # rules line at x=0
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted') # not sure what this does
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$') # maths phi

# %%




