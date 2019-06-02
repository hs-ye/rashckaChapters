# -*- coding: utf-8 -*-
"""
Created on Nov 08 07:33:40 2017

Raschka - Python Machines Learning
Follow along

Chapter 4 - Data preprocessing

@author: yehan
"""

import pandas as pd
import numpy as np
from io import StringIO

np.set_printoptions(threshold=np.inf)  # prints the full np matricies (repr), not truncated versions
np.set_printoptions(threshold=1000)  # sets limit before np summaries are printed
# %% csv missing data

# note cannot have spaces in a CSV file, it'll mess up type casting (thinks numeric is str)
# having a single space in a cell casts entire row to string
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0, ,8.0
6,,9.0,10.5
0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))  # Note we are using pandas
df

df.isnull()
df.isnull().sum()  # goes across columns
df.values  # Note: provides numpy array access to the dataframe

df.dropna()  # note that row 2 col 3 is a space, which is an acceptable value
df.dropna(axis=1, how='any')  # axis is row (0) or col (1), how is 'any' or 'all'
df.dropna(axis=1, how='all')  # for 'all', must be all missing to be dropped
df.dropna(thresh=3)  # thresh=number of non-NAN values in row, to keep it
df.dropna(thresh=4)  # rows (or cols, if axis=1) with less than this will be dropped
df.dropna(thresh=4, axis=1)  # cols

df.dropna(subset=['D'])  # only check column D

# %% Mean imputation

df2 = df.replace(' ',np.nan)  # fill holes with empty
df2['C'] = df['C'].astype(float)  # astype doesn't work for nans...
df2['C'] = pd.to_numeric(df2['C'])  # built in pd method to convert with nan handling

from sklearn.preprocessing import Imputer

wrk_imp = Imputer(missing_values='NaN', strategy='mean', axis=0)  # all default params
wrk_imp = wrk_imp.fit(df2)  # note the caps matters

imputed_data = wrk_imp.transform(df2.values)


# %% Categorical Data
df_cat = pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']])

df_cat.columns = ['color', 'size', 'price', 'classlabel']

size_map = {'XL':'3',
			'L':'2',
			'M':'1'}

df_cat.size = df_cat.size.map(size_map)  # doesn't work, size is attr of df
df_cat['size'] = df_cat['size'].map(size_map)  # applies map to a col

inv_size_map = {v: k for k, v in size_map.items()}  # dict comprehension
#inv_size_map = [(v,k) for k, v in size_map.items()]  # list comprehension ver.
# %% categorical labels
df_cat
class_map = {v:k for k, v in enumerate(df_cat['classlabel'].unique())}
df_cat['classlabel'] = df_cat['classlabel'].map(class_map)
df_cat
# reverse the transformation
class_map_inv = {v:k for k,v in class_map.items()}
df_cat['classlabel'] = df_cat['classlabel'].map(class_map_inv)
df_cat

# or use the built in encoder...?
from sklearn.preprocessing import LabelEncoder
wrk_lab = LabelEncoder()
y = wrk_lab.fit_transform(df_cat['classlabel'].values)
y
wrk_lab.inverse_transform(y)  # built in reverse transformation

# %% Use the encoder to do ordinal encoding

X = df_cat[['color', 'size', 'price']].values
wrk_lab = LabelEncoder()  # this class only works for 1 column at a time
				  # also produces ordinal encodings, which we may not want
X[:,0] = wrk_lab.fit_transform(X[:,0])  # can't do slices, like [0:1], not allowed
#X = wrk_lab.fit_transform(X)  # doesn't work, even if all are string categories

# %% dummy/'one hot' encoding
from sklearn.preprocessing import OneHotEncoder
# param is to control which column to encode
wrk_ohe = OneHotEncoder(categorical_features=[0])
wrk_ohe.fit_transform(X).toarray()  # gives np array (dense), to check visually
# default outputs to sparse array, more efficient for actual/large data

# or better way to do dummies
pd.get_dummies(df_cat[['price','color','size']])  # wow that was easy

# %% Next step - test & training sets
# get Wine data
from sklearn.datasets import load_wine  # needs scikit-learn v19
#df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
#				  header=None)
# only in scikit-learn v18+ onwards, data is built in
df_wine = pd.DataFrame(load_wine().data)  # only gives data, not the y-value labels
a = load_wine()
load_wine().target

df_wine.columns = load_wine().feature_names
df_wine['Class labels'] = load_wine().target
print('Class labels', df_wine['Class labels'].unique())
df_wine.head()

# %% test train split

from sklearn.model_selection import train_test_split as tts  # moved from sklearn.cross_validation

X, y = df_wine.drop(['Class labels'], axis=1).values, df_wine.loc[:,'Class labels'].values
X_train, X_test, y_train, y_test= \
	tts(X, y, test_size=0.3, random_state=0)

# %% rescaling: normalising vs standardising
# Normalise:
from sklearn.preprocessing import MinMaxScaler  # scales things to between 0 and 1
wrk_mms = MinMaxScaler()
X_train_norm = wrk_mms.fit_transform(X_train)
X_test_norm = wrk_mms.transform(X_test)

# extra: use PD dataframes to describe
pd.DataFrame(X_train_norm).describe()
pd.DataFrame(y_train).describe()

# Standardised
from sklearn.preprocessing import StandardScaler
wrk_std = StandardScaler()
X_train_std = wrk_std.fit_transform(X_train)
X_test_std = wrk_std.transform(X_test)

# %% Regularisation
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')
mod_lrm_l1 = LogisticRegression(penalty='l1', C=0.1)
mod_lrm_l1.fit(X_train_std, y_train)
mod_lrm_l1.score(X_train_std, y_train)  # train acc
mod_lrm_l1.score(X_test_std, y_test)  # test acc

# intercept and coeffs
mod_lrm_l1.intercept_  # one vs rest regression, intercepts for class 1, 2 and 3
mod_lrm_l1.coef_  # as above, 3 sets of coefs

# %% Regularisation path - weight coefficients with changing regularisation penalty
import matplotlib.pyplot as plt
fig = plt.figure(2)
ax = plt.subplot(111)

# need a colour for each X feature, track it through changing iterations
colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
		   'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights, params = [], []  # set up lists to append to

for c in range(-4, 6):
	mod_lrm_l1 = LogisticRegression(penalty='l1', C=10**c, random_state=0)
	mod_lrm_l1.fit(X_train_std, y_train)
	weights.append(mod_lrm_l1.coef_[1])
	params.append(10**c)

weights = np.array(weights)
# make a iterator of color and the column number
for column, colour in enumerate(colours):
	plt.plot(params, weights[:, column],
			 label=df_wine.columns[column],
			 color=colour) # column
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**(5)])  # note lower limit is 10^-5 >0
plt.xscale('log')
plt.ylabel('weight coefs')
plt.xlabel('C')
plt.legend(loc='upper left')
# next line supposed to make a second box to the left, but not working atm...
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()

# %% Sequential Backward Selection implementation

from sklearn.base import clone  # makes a deep copy of estimator w/o data
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split  # will be removed soon
from sklearn.metrics import accuracy_score

class SBS():
	def __init__(self, estimator, k_features, scoring=accuracy_score,
		    test_size=0.25, random_state=1):
		'''initialise inputs required, which are:'''
		self.scoring = scoring  # scoring method, this is a function (see class params)
		self.estimator = clone(estimator)  # a copy of the estimator passed in
		self.k_features = k_features  # num of target features
		self.test_size = test_size  # for the training and validation split, not actually 'test'
		self.random_state = random_state  # seed initialisation

	def fit(self, X, y):
		''' goes through all combinations of features, and tracks their performance against the
		chosen measure, saves the best ones at each number of feature
		'''
		X_train, X_test, y_train, y_test = \
			train_test_split(X, y)  # this is validation set, no test data used for fitting
		dim = X_train.shape[1]  # columns of X
		self.indicies_ = tuple(range(dim))  # makes a tuple length = # cols of input
		self.subsets_ = [self.indicies_]
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indicies_)
		self.scores_ = [score]
		# loop to keep eliminating features until desired number reached
		while dim > self.k_features:
			scores = []  # save score of feature set
			subsets = []  # save the set of features that generated the score

			# makes combinations of elemnents with length(n features-1)
			# results in all combinations of features, leaving 1 out at a time
			for p in combinations(self.indicies_, r=dim-1):
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)  # get the score of each combination of features
				subsets.append(p)  # append the set of features that got the score

			best = np.argmax(scores)  # find the position of max score
			self.indicies_ = subsets[best]  # returns max score at this combo.
			'''Note: The last time this is set to be set will be the best performing combination at
			For the given number of features to be selected by this stepwise algorithm
			Not necessarily the best across the num of features though
			'''
			self.subsets_.append(self.indicies_)  # best features for each #features in loop
			dim -= 1

			self.scores_.append(scores[best])
		self.k_score_ = self.scores_[-1]
		return self  # object holding thebest scores etc

	def transform(self, X):
		'''Selects only the columns of X which have been fitted'''
		return X[:, self.indicies_]

	def _calc_score(self, X_train, y_train, X_test, y_test, indicies):
		'''given subset of data, used to calculate score based on model'''
		self.estimator.fit(X_train[:, indicies], y_train)
		y_prd = self.estimator.predict(X_test[:, indicies])
		score = self.scoring(y_test, y_prd)
		return score


# %% Apply stepwise backwards selection to the KNN

from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt

mod_knn = KNN(n_neighbors=2)
sbs = SBS(mod_knn, k_features=1)  # do all fits up to only 1 feature
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
# plt.show()

# list of features that gave good prediction
k5 = list(sbs.subsets_[8])
print(df_wine.columns[k5])

# performance of original full model on test set
mod_knn.fit(X_train_std, y_train)
print('Trainning acc:', mod_knn.score(X_train_std, y_train))
print('Test Acc:', mod_knn.score(X_test_std, y_test))

# performance of reduced features selected
mod_knn.fit(X_train_std[:, k5], y_train)  # fit model on reduced feature set
mod_knn.score(X_train_std[:, k5], y_train)  # train acc of reduced sample
mod_knn.score(X_test_std[:, k5], y_test)  # test acc of reduced sample

# %% Feature importance: Random Forests (only)

from sklearn.ensemble import RandomForestClassifier as RFC

x_labels = df_wine.columns[:-1]  # last column is y
mod_rfc = RFC(n_estimators=10000, random_state=0, n_jobs=-1)  # this will take a while
mod_rfc.fit(X_train, y_train)
importances = mod_rfc.feature_importances_
importance_rank = np.argsort(importances)[::-1]  # sorts, then reverses the output
# look up python's extended slices to see how this works. 3rd is steps argument, negative reverses
for f in range(X_train.shape[1]):
	print("{0:2}) {1:<35} {2:.5}".format(
			f, x_labels[f], importances[importance_rank[f]]))

plt.figure(2)
plt.title('Feature Importances for Random Forest')
plt.bar(range(X_train.shape[1]), importances[importance_rank],  #can slice ndarray with ndarray
		color='darkred', align='center')
plt.xticks(range(X_train.shape[1]), x_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()  # moves plot to show labels properly
#plt.show()


# %% Extra stuff - check how np.argmax vs. max works

import random

a = sorted(range(10), key=lambda x: random.random())  # random shuffled list
# input values are sorted by the key param, which passes each element in the
# input list to the lambda function, which then ranks them
# in this case, they will all be random

max(a)  # returns max value
np.argmax(a)  # returns positoin of max value

