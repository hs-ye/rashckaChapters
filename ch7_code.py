# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 23:55:34 2017
Raschka - Python Machines Learning
Follow along

Chapter 7 - Ensembles
@author: yehan
"""

from scipy.misc import comb
import math
import numpy as np
import pandas as pd  # not required in book, convenience
import matplotlib.pyplot as plt

# %%
def ensemble_error(n_classifier, error):
	k_start = math.ceil(n_classifier / 2.0)  # number of votes to win
	probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k)
										for k in range(k_start, n_classifier + 1)]
	# makes a list of the probabilty of landing on each possible number of outcomes
	return sum(probs)  # sum of the PMF for the parts we care about to get the error

ensemble_error(n_classifier=11, error=0.25)

# %%  Plot the total error for a range of base errors

err_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in err_range]

plt.figure(0)
plt.plot(err_range, ens_errors, label='Ensemble error', linewidth=2)
plt.plot(err_range, err_range, label='Base error', linestyle='--', linewidth=2)
plt.xlabel('Base error rate')
plt.ylabel('Actual error')
plt.legend(loc='best')
plt.grid()
#plt.show()

# %% example of calculating voting outcomes with weights
np.bincount([0, 0, 1])  # gives [2, 1], totals 2 for class 0, 1 for class 1
np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])  # weight on each element totals 40% for 0, 60% for 1

np.argmax(  # returns indicies of the max values, along an axis
		np.bincount([0, 0, 1],  # counts how many occurences of integers from 0 to the max value in the array
		weights=[0.2, 0.2, 0.6])  # weights each outcome by a count
	)
# so what this is doing is, taking in weighted votes, and returning the highest index of the
# weighted outcome

# %% Weighted majority vote based on class probabilities
ex = np.array([[0.9, 0.1],
				[0.8, 0.2],
				[0.4, 0.6]])
np.average(ex, axis=0)    # avg along cols
np.average(ex, axis=1)    # avg across rows, is 0.5 for everything, by definition
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])    # avg across rows, weighted
print(p)

np.argmax(p)  # shows which class has higher pr, after totalling all weighted pr of each

# %%  implement a majorityVoteClassifier, by hand - appears to work now?
from sklearn.base import BaseEstimator  # these are the base class templates for new classifiers
from sklearn.base import ClassifierMixin  # A Mixin is an interface
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six  # needed for py2.7 compatibility
from sklearn.base import clone
from sklearn.pipeline import _name_estimators  # makes names for estimators??
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
	''' A majority Vote ensemble classifier

	Params:
	-----------
	classifiers: array-like, shape = [n_classifiers]
	different classifiers for the ensemble

	vote : str, {'Classlabel', 'probability'}, one or the other pls
	Default: 'Classlabel'
	If 'classlabel' the prediction is based on the
	argmax of class labels, Else if given list of predicted 'probability',
	the  argmax of the sum of probabilities is used as predictoin
	(recommended for calibrated classifiers).

	weights : array-like, shap = [n_classifiers]
	Optional, default: None
	If a list of int or float values are provided, the classifiers are
	weighted by importances; uniform if none provided
	'''
	def __init__(self, classifiers, vote='classlabel', weights=None):
		self.classifiers = classifiers
		self.named_classifiers = {key:value for key, value in
					   _name_estimators(classifiers)}  # gets name of classifiers
		self.vote = vote
		self.weights = weights

	def fit(self, X, y):
		""" Fit classifiers.

		Params:
		-----------
		X : {array-like, sparse matrix},
			shape = [n_samples, n_features]
			Matrix of training samples.

		y : array-like, shape = [n_samples]
			Vector of target class labels.

		Returns
		----------
		self : object
		"""
		# Use Label encoder to ensure class laabels start with 0, needed for np.argmax
		self.labelenc_ = LabelEncoder()
		self.labelenc_.fit(y)  # do the label transform
		self.classes_ = self.labelenc_.classes_  # got the classes now from the labelenc obj
		# call in self.predict
		self.classifiers_  = []
		for clf in self.classifiers:
			fitted_clf = clone(clf).fit(  # clone to get copy of clf
										X,
										self.labelenc_.transform(y)
									)
			self.classifiers_.append(fitted_clf)
		return self

	def predict(self, X):
		""" Predict class labels for X, given either input votes of probabilities,
		or if prediciton outcomes are class labels

		Params:
		----------
		X : {array-like, sparse matrix}, Shape = [n_samples, n_features]
			Matrix of training samples

		Returns:
		----------
		maj_vote : array-like, shape = [n_samples], predicted class labels

		"""
		if self.vote == 'probability':
			maj_vote = np.argmax(self.predict_proba(X), axis=1)  # predict_proba via inheritance
		else:  # 'classlabel' vote, instead of probability
			# Corllect results from clf.predict calls to whatever the object has been fitted with
			predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
			maj_vote = np.apply_along_axis(
					lambda x: np.argmax(np.bincount(x, weights=self.weights)),
					axis=1,
					arr=predictions
				)
		maj_vote = self.labelenc_.inverse_transform(maj_vote)  # returns to original
		return maj_vote

	def predict_proba(self, X):
		""" Predict class probabilities for x, after training

		Params:
		----------
		X : {array-like, sparse matrix}, Shape = [n_samples, n_features]
			Training vectors - n_sample is training rows, n_features is columns

		Returns:
		----------
		avg_proba : array-like, shape = [n_samples, n_classes],
		produces 1 weighted avg predicted class prob for each class, per input row

		"""
		probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
		avg_proba = np.average(probas, axis=0, weights=self.weights)
		return avg_proba

	def get_params(self, deep=True):
		''' get Classifier param names for GridSearch,
		loop through the classifiers and get the combination of classifier name and the
		params they have, e.g. lrm__C for regularisation
		'''
		if not deep:
			return super(MajorityVoteClassifier, self).get_params(deep=False)
		else:
			out = self.named_classifiers.copy()
			for name, step in six.iteritems(self.named_classifiers):
				for key, value in six.iteritems(step.get_params(deep=True)):
					out['{0}__{1}'.format(name, key)] = value
					#  will be used for programmatic hyperparam tuning
			return out

# %% load iris data - will use to test the majority classifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
# pd.get_dummies(y)  # pd version, makes 2 separate cols
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
											X,
											y,
											test_size=0.5,
											random_state=1
										)

# %% Train 3 separate classifiers
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(C=0.001, random_state=0)  # default penalty='l2'
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2)  #default metric='minkowski'

# need pipelines for lrm and knn to standardise inputs, not necessary for tree
pipe_lrm = Pipeline([
			['sc', StandardScaler()],
			['clf', clf1]
		])
# have separate pipes, so the 'clf' object can have the same name and be referenced same way
pipe_knn = Pipeline([
			['sc', StandardScaler()],
			['clf', clf3]
		])

clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']  # dunno why these are separate
print('10-fold cross validation:\n')

for clf, label in zip([pipe_lrm, clf2, pipe_knn], clf_labels):
	scores = cross_val_score(
			estimator=clf,
			X=X_train,
			y=y_train,
			cv=10,
			scoring='roc_auc'
		)
	print("ROC AUC: {0:.2} (+/- {1:.2}) [{2}]".format(scores.mean(), scores.std(), label))

# %% implement the ensemble of the 3 classifiers using CV scoring

#from sklearn.ensemble import VotingClassifier as MajorityVoteClassifier  # sk implementation

mv_clf = MajorityVoteClassifier(classifiers=[pipe_lrm, pipe_knn, clf2])  # use manual version
#mv_clf = MajorityVoteClassifier(
#		estimators=[('lrm',pipe_lrm), ('knn', pipe_knn), ('tree', clf3)],
#		voting='soft'
#	)  # use sklearn version
clf_labels += ['mv ensemble']
all_clf = [pipe_lrm, clf2, pipe_knn, mv_clf]

for clf, label in zip(all_clf, clf_labels):
	scores = cross_val_score(
		estimator=clf,
		X=X_train,
		y=y_train,
		cv=10,
		scoring='roc_auc' # accuracy doesn't work here for some reason, TBD
	)
	print("AUC {0:.2}, (+/- {1:.2}) {2}".format(
			scores.mean(),
			scores.std(),
			label
		))

# %% evaluate and tuning the ensemble

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

colours = ['black', 'red', 'green', 'blue']
linestyles = [':', '-', '-.', '--']

for clf, label, clr, ls in zip(all_clf, clf_labels, colours, linestyles):
	# assume label of positive class is 1
	y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
	# save values needed to plot roc_curves
	fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
	# also calculate the AUC from the roc co-ordinates
	roc_auc = auc(x=fpr, y=tpr)
	plt.plot(fpr, tpr, color=clr, linestyle=ls,
		label='{0} (auc= {1:.2})'.format(label, roc_auc))
	plt.legend(loc='lower right')
	plt.plot([0, 1], [0, 1], linestyle='--', color='grey',linewidth=2)
	plt.xlim([-0.1, 1.1])
	plt.ylim([-0.1, 1.1])
	plt.grid()
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# plt.show()

# %% visualise the decision boundary area, for 2 features
from itertools import product  # element wise multiplication

tfm_sc = StandardScaler()  # normalise scale for visualisation
X_train_std = tfm_sc.fit_transform(X_train)

x_min = X_train_std[:, 0].min() - 0.5  # first variable is x
x_max = X_train_std[:, 0].max() + 0.5
y_min = X_train_std[:, 1].min() - 0.5  # 2nd variable is y
y_max = X_train_std[:, 1].max() + 0.5

# set up all grid of values that will be plotted, xx and yy are actually x1 and x2, both inputs
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
	np.arange(y_min, y_max, 0.1))
# set up subplots, which are held in the 'axarr' (??) objects
f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))
# do actuall fitting and graphing
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
	clf.fit(X_train_std, y_train)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # predict over all values of x1 and x2
	Z = Z.reshape(xx.shape)  # get the predictions into shape
	# for all 4 subplots that have been set up, plot overall decision space
	axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
	# plot the predictions of the points in each subplot
	axarr[idx[0], idx[1]].scatter(  # when actual y is 0
		X_train_std[y_train == 0, 0],
		X_train_std[y_train == 0, 1],
		c='blue',
		marker='^',
		s=50
	)
	axarr[idx[0], idx[1]].scatter(  # when actual y is 1
		X_train_std[y_train == 1, 0],
		X_train_std[y_train == 1, 1],
		c='red',
		marker='s',
		s=50
	)
	axarr[idx[0], idx[1]].set_title(tt)


# %% methods to tune a ensemble, the parts of it and the meta-classifier

from sklearn.grid_search import GridSearchCV

mv_clf.get_params()  # gets the names of the params for a grid search
# run the above, then see below on how to use

params = {
		'decisiontreeclassifier__max_depth': [1, 2],
		'pipeline-1__clf__C': [0.001, 0.1, 100.0]
	}

grid = GridSearchCV(
		estimator=mv_clf,
		param_grid=params,
		cv=10,
		scoring='roc_auc'
	)

grid.fit(X_train, y_train)

# loops through all the possible combination of the grid search and the score attained
for params, mean_score, scores in grid.grid_scores_:
	print("{0:5.2}+/-{1:5.2}  {2}".format(mean_score, scores.std() / 2, params))

print('Best parameters {0}'.format(grid.best_params_))
print('Best AUC: {0}'.format(grid.best_score_))

# %% TODO - the part on Bagging, skipped for now

