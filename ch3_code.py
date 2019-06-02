# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 07:33:40 2017

Raschka - Python Machines Learning
Follow along

Chapter 3

@author: yehan
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Load/inspect data
from sklearn import datasets

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

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, fig=0):
	'''Testing variables:
	classifier = Perceptron()
	classifier.fit(X, y)
	resolution=0.02
	'''
	# added extra param to allow for multiple flots
	plt.figure(fig)
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
X_combined_std = np.vstack((X_train_std, X_test_std))
y_all = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_all,
					  classifier=ppn, test_idx=range(105, 150), fig=0)
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
plt.show()  # not necessary for ipython

# %% logistic regression with scikit-learn

from sklearn.linear_model import LogisticRegression

mod_lrm = LogisticRegression(C=1000.0, random_state=0)  # model: Logistic regression model
mod_lrm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_all, classifier=mod_lrm, test_idx=range(105, 150), fig=1)
plt.xlabel('petal length [std.]')
plt.ylabel('petal width [std.]')
plt.legend(loc='lower center')
plt.show()  # not necessary for ipython

mod_lrm.predict_proba(X_test_std[0,:].reshape(1, -1))  # gives the prd. probabilities for each class

# %% applying regularisation to logistic regression

weights, params = [], []
for c in range(-5, 5):  # cant use np.arange here, which returns np.ints, can't be raised to negative powers
	mod_lrr = LogisticRegression(C=10**c, random_state=0)  # model: log regression w/ regularsation
	mod_lrr.fit(X_train_std, y_train)
	print(mod_lrr.coef_)  # for 3 class output, 3 models get fitted with logistic reg.
	# to simulate 3x 1 vs all models. good interview question
	weights.append(mod_lrr.coef_[1])  # only gets coeff for the model of the 2nd class
	params.append(10**c)  # ** is ^, raising to power

weights = np.array(weights)  # converts array of tuples into array of lists, which can be sliced
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.legend(loc='lower left')

# %%

from sklearn.svm import SVC

mod_svm = SVC(kernel='linear', C=1, random_state=0)
mod_svm.fit(X_train_std, y_train)

sum(mod_svm.predict(X_test_std)==y_test)/len(y_test)  # accuracy

plot_decision_regions(X_combined_std, y_all, classifier=mod_svm, test_idx=range(105,150),fig=2)
plt.xlabel('petal len')
plt.ylabel('petal wid')
plt.legend(loc='bottom left')  # doesn't work, but gives you actual possible choices
plt.legend()
plt.axis('off')


# %% Stochastic GD classifiers - use the .partial_fit() method
from sklearn.linear_model import SGDClassifier

ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')


# %% non linear SVM - make fake data
np.random.seed(0)
X_xor = np.random.randn(200, 2)  # random array of 200 rows x 200 cols
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)  # used for comparing multiple arrays elementwise
y_xor = np.where(y_xor, 1, -1)  #converts a bool array into a -1 or 1 array

plt.figure(1)  # used to choose which figure/plot to draw on
plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], color='blue', marker='x', label='1')  # plot 1's
plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1], color='red', marker='s', label='1')  # plot 1's
plt.ylim(-3,3)
plt.legend()


# %% RBF/Gaussian Kernel for SVM on non-linear values
from sklearn.svm import SVC
mod_svm_rbf = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)  # rbf is acutally default kernel
# gamma is param controlling similarity measure of kernel
mod_svm_rbf.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=mod_svm_rbf, fig=2)
plt.legend()

# %% SVM on Iris data
mod_svm_iris = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
mod_svm_iris.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_all, mod_svm_iris, test_idx=range(105, 150), fig=3)
plt.legend()

# %% SVM modified gamma param

mod_svm_iris = SVC(kernel='rbf', random_state=0, gamma=100, C=1.0)
mod_svm_iris.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_all, mod_svm_iris, test_idx=range(105,150), fig=4)
plt.legend()


# %% Trees - Node Cost functions
# %reset  # ipython magic function to clear workspace

import numpy as np
import matplotlib.pyplot as plt
def gini(p):
	return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))

def entropy(p):
	return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def error(p):  # misclassification
	return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01) # make a vector from 0 to 1 with steps 0.01
#range(0.0, 1.0, 0.01))  # doesn't work with floats
ent = [[p, entropy(p)] if p != 0 else None for p in x]  # floating point maths is broken lol
ent = [entropy(p) if p != 0 else None for p in x]  # floating point maths is broken lol
sc_ent = [e * 0.5 if e else None for e in ent]  # scaled version of entropy
err = [error(i) for i in x]  # misclassification rate

# make a plot of all the stuff
fig = plt.figure(5)
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
							 ['Entropy', 'Entropy (scaled)', 'Gini', 'Misclassification'],
							 ['-', '-', '--', '-.'],
							 ['black', 'lightgray', 'red', 'green', 'cyan']):
	line = ax.plot(x, i ,label=lab, linestyle=ls, lw=2, color=c)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
	ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
	ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
	plt.ylim([0, 1.1])
	plt.xlabel('p(i=1)')
	plt.ylabel('Impurity index')
	#plt.show()


# %% decision trees

from sklearn.tree import DecisionTreeClassifier

mod_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
mod_tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))  # works on 1d arrays (i.e. no explicit columns)
#y_combined = np.vstack((y_train.reshape(-1, 1), y_test.reshape(-1, 1)))  # same as above
# except has an explicit column
plot_decision_regions(X_combined, y_combined, classifier=mod_tree,
						 test_idx=range(105,150), fig=8)
plt.xlabel('petal length cm')
plt.ylabel('petal width cm')
plt.legend(loc='lower right')
plt.title('Iris flowers - Tree classifier')


# %% Extra - tree plotting
from sklearn.tree import export_graphviz

# Note graphviz
export_graphviz(mod_tree, out_file='tree.dot',
			 feature_names=['petal length', 'petal width'])



# %% Aside/extra stuff

# np.int can't do negative powers

%timeit for x in range(int(-5E5), int(5E5)): x ** 2
%timeit for x in np.arange(int(-5E5), int(5E5)): x ** 2
# np won't work
%timeit for x in range(-5000, 5000): x ** x
%timeit for x in np.arange(-5000, 5000): x ** x


# fit logistic regression on random binary classes
import random as rand

y_rng = [rand.choice([-1, 1]) for _ in range(0, 105)]

mod_rand = LogisticRegression(C=10, random_state=0)
mod_rand.fit(X_train_std, y_rng)
mod_rand.coef_  # should only have 1 pair of coefs.
plot_decision_regions(X_train_std, y_rng, mod_rand)


# Plot using extra utility
from adspy_shared_utilities import plot_decision_tree












