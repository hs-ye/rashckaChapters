# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:43:20 2017
Raschka - Python Machines Learning
Follow along

Chapter 6 - Model Evaluation & Hyperparameter Tuning (Cross validation)
@author: yehan
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold

# %% load data
# you could get it from internet - includes a patient ID 1st column
'''
df = pd.read_csv('https://archive.ics.uci.edu/ml/\
machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
header=None)

X = df.loc[:, 2:].values  # if loading from internet location
y = df.loc[:, 1].values
from sklearn.preprocessing import LabelEncoder

tfm_le = LabelEncoder()  # don't need this step for local load
y = tfm_le.fit_transform(y)
tfm_le.transform(['M', 'B'])  # demo of what it's doing, nothing done.

#tfm_le.transform(['M', 'B', 'b'])  # ERROR: case sensitive
#y_map = {"M": 1, "B": 0}  # note if you don't define a map, it will go to NaN

'''
# from local sklearn library, no ID column included
r_data = load_breast_cancer()
df = pd.DataFrame(
		data=np.c_[r_data.target.reshape(-1,1), r_data.data],  # column stack
		columns=['Class labels'] + r_data.feature_names.tolist())
r_data.target_names

X = df.iloc[:, 1:].values  # if loading from local
y = df.iloc[:, 0].values
y_map = {1: 0, 0: 1}  # swich 1's and 0's, definition is different from web source
y = pd.Series(y).map(y_map)



# %% split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =\
	train_test_split(X,  y, test_size=0.2, random_state=1)

# %% Pipeline demo - set up basic pipeline object (run before next one)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([
		('scl', StandardScaler()),
		('pca', PCA(n_components=2)),
		('mod_lrm', LogisticRegression(random_state=1))
	])

pipe_lr.fit(X_train, y_train)
pipe_lr.score(X_test, y_test)  # prints accuracy after being fit
pipe_lr.predict(X_test)  # does predictions using all previous steps


# %% how to set up k-fold CV, stratified
''' # cross_validation ver - this has been deprecated, interface is difference
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
# the constructor used to create a generator object...different below
'''

from sklearn.model_selection import StratifiedKFold  # new interface, makes dedicated object

tfm_skf = StratifiedKFold(n_splits=10, random_state=1)  # new API, makes an object first
# tfm_skf.get_n_splits(y=y_train)  # spits out the n_splits param attached to the object
kfold = tfm_skf.split(X=np.zeros(y_train.shape), y=y_train)  # the method needed
# creates the same generator that does what the old API used to do directly

''' note on generators, (kfold is a generator)
next(kfold)  # loading 1 generator object at at time. Generators can't be rewound
list(kfold)  # if you run the above, then this
'''

scores = []
'''
# CAUTION somehow the train test split screws up the index for y_train
# the index number for subset/slicing is what kfold generates, so won't work unless
# index is reset
'''
y_train_no_index = y_train.reset_index(drop=1)  # make a copy of y without the index
for k, (train,test) in enumerate(kfold):  # old cross_validation ver.
	pipe_lr.fit(X_train[train], y_train_no_index[train])
	score = pipe_lr.score(X_train[test], y_train_no_index[test])
	scores.append(score)
	print(" Fold {0}: Class dist = {1}; Acc {2}%".format(
				k + 1,
				np.bincount(y_train_no_index[train]),
				score*100
		))  # tested works now

# %% k fold CV scoring using built in method - apply to an estimator object
# or one like pipeline which has a .fit() function the same way that estimators do

#from sklearn.cross_validation import cross_val_score  # old
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print(scores)
print('cv mean scores: ', np.mean(scores), '+/-', np.std(scores))

# should give same results as above block

# %% learning curves - performance vs. n samples.
#from sklearn.learning_curve import learning_curve  # old location, moved
from sklearn.model_selection import learning_curve  # apply to an estimator object

pipe_lr = Pipeline([
	('scl', StandardScaler()),
	('clf', LogisticRegression(penalty='l2', random_state=0))])

# LRM estimator with sizes, then train/test scores for each size sample tried
train_sizes, train_scores, test_scores = learning_curve(
		estimator=pipe_lr,
		X=X_train,
		y=y_train,
		train_sizes=np.linspace(0.1, 1.0, 10),  # stepping for % of total data used. see help
		cv=10,  # folds for cross validation for each size fitted, param has other options
		n_jobs=1
	) # returns the size used for each step
# At each step, the scores for each of the folds, which we can then take average

train_mean = np.mean(train_scores, axis=1)  # rows is fold, col is by step size
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# %% Plot
plt.figure(0)
plt.plot(train_sizes, train_mean, color='b', marker='o', label='avg. training')  #
plt.plot(train_sizes, test_mean, color='g', marker='^', label='avg. validation',
		 linestyle='--')
plt.ylim([0.85, 1])
plt.ylabel('Accuracy')
plt.xlabel('N Training samples')
plt.legend()
plt.grid()

plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
				 alpha=0.15, color='blue')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
				 alpha=0.15, color='green')

# %% validation curves
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
		estimator=pipe_lr,
		X=X_train,
		y=y_train,
		param_name='clf__C',  # this is saying go to the object inside the estimator
		# called clf, and find the param C. Specific to pipeline objects
		param_range=param_range,
		cv=10
	)

train_mean = np.mean(train_scores, axis=1)  # rows is fold, col is by step size
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# %% Plot validation curves

plt.figure(1)
plt.plot(param_range, train_mean, color='b', marker='o', label='avg. training')
plt.plot(param_range, test_mean, color='g', marker='^', label='avg. test')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std,
				 alpha=0.15, color='blue')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std,
				 alpha=0.15, color='green')

plt.ylim([0.85, 1])
plt.ylabel('Accuracy')
plt.xlabel('Param C')
plt.legend()
plt.xscale('log')
plt.grid()

# %%

#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [10**e for e in range(-4, 4)]
# two sets of params to search through, for a linear one and a rbf one
param_grid = [{'clf__C': param_range,
			   'clf__kernel': ['linear']},
			  {'clf__C': param_range,
			   'clf_gamma': param_range,
			   'clf_kernel': ['rbf']}]

gs = GridSearchCV(
		estimator=pipe_svc,
		param_grid=param_grid,
		scoring='accuracy',
		cv=10,
		n_jobs=-1  # max cores possible
	)

# %% run and get results
gs=gs.fit(X_train, y_train)
print(gs.best_score_, ' ', gs.best_params_)

clf = gs.best_estimator_  # get the params of the best estimator
clf.fit(X_train, y_train)  # apply them to data
print('test accuracy: {:3.3}'.format(clf.score(X_test, y_test)))

# %% nested cross validation, using the SVC pipe to do
# just put the grid search inside another object/funciton that does CV folds
scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5, n_jobs=-1)
print('Mean acc.: {:3.3} std dev: {:3.3}'
	.format(np.mean(scores), np.std(scores)))

# %% do nested CV on a decision tree to compare performance
from sklearn.tree import DecisionTreeClassifier as dtree

gs_tree = GridSearchCV(
		estimator=dtree(random_state=0),
		param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
		scoring='accuracy',
		cv=5,
		n_jobs=-1
	)

scores_tree = cross_val_score(
		estimator=gs_tree,
		X=X_train,
		y=y_train,
		scoring='accuracy',
		cv=5,
		n_jobs=-1
	)
print('Mean acc.: {:3.3} std dev: {:3.3}'
	.format(np.mean(scores_tree), np.std(scores_tree)))


# %% Confusion matrix

from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_test, y_pred)
print(confmat)
# getting the confusion matrix by hand
sum(y_test[y_pred==False]==True)  # where the prediction is 0, and test is 1
sum(y_test[y_pred==True]==True)  # where the prediction is 1, and test is 1
sum(y_test[y_pred==False]==False)  # where the prediction is 0, and test is 0
sum(y_test[y_pred==True]==False)  # where the prediction is 1, and test is 0


# %% make the conf matrix pretty
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
	for j in range(confmat.shape[1]):
		ax.text(x=j,  # for the row
				y=i,  # for each col
				s=confmat[i, j],  # text is whatever it says in the same place of confmat
				va='center',
				ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
#plt.show()

# %% scoring metrics
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

# %% Making scorers objects for use in grid search
from sklearn.metrics import make_scorer
scorer = make_scorer(score_func=f1_score, pos_label=0)  # makes a callable score fn
# use in pipelines, estimators, grid search, cross validation objects etc.

# %% Plot ROC curve
from sklearn.metrics import roc_curve, auc
from numpy import interp
from sklearn.model_selection import StratifiedKFold  # new API class for this
# from sklearn.cross_validation import StratifiedKFold  # Old API, deprecated

# note use of new API for StratifiedKFold
X_train2 = X_train[:, [4, 14]]  # select 2 columns from X_train, as prev defined
tfm_skf = StratifiedKFold(n_splits=3, random_state=1)  # new stratifiedKFold object
cv = list(tfm_skf.split(X = X_train2, y=y_train))  # Note X data is actually not used. Just used for shape
# the above actually gives a generator object, some methods in this code assumes
# it's a list
fig = plt.figure(num=2, figsize=(7,5))
mean_tpr = 0.0  # true pos rate
mean_fpr = np.linspace(0, 1, 100)  # false pos rate
all_tpr = []

for i, (train, test) in enumerate(cv):  # each item in CV is a list of train/test row IDs
	#print(train, test, i)  # makes the train, test obs indicies
	print(i, '\n', mean_tpr)
	probas = pipe_lr.fit(
		X_train2[train],
		y_train.iloc[train]  # note: how to use an index mask on a scrambled index
	).predict_proba(X_train2[test])  # probs - 2 cols, 1 for each class, adds to 1
	# get the roc curve - produces pairs of stuff
	fpr, tpr, thresholds = roc_curve(
			y_true=y_train.iloc[test],
			y_score=probas[:, 1],
			pos_label=1
		)   #fpr is x, tpr is y co-ords. Thesholds is for each co-ord
#	print(fpr, '\n')
#	print(tpr, '\n')
#	print(mean_fpr, '\n')
	# linear interpol: put in X-y points to interpolate between (the tpr and fpr)
	# then ofr a list of x - it iwll give you y the sits on the line between x-y
	mean_tpr += interp(mean_fpr, fpr, tpr)   # add raw outputs, divide later
	mean_tpr[0] = 0.0  # first point bottom left
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr,
		    tpr,
		    lw=1,
		    label='ROC fold {} (area = {:0.2})'.format(i+1, roc_auc))
# draw diagonal random flipping line
plt.plot([0, 1],
	    [0, 1],
	    linestyle='--',
	    color=(0.6, 0.6, 0.6),
	    label='random guessing')
# divide  mean_tpr by 3 because we added 3x roc curves
mean_tpr /= len(cv)  # same as mean_tpr = mean_tpr / len(cv)
mean_tpr[-1] = 1.0 # setting the last value to be 1, in the corner
mean_auc = auc(mean_fpr, mean_tpr)
# plot the mean roc curve
plt.plot(mean_fpr, mean_tpr, 'k--',
	    label='Mean ROC (area= {:0.2})'.format(mean_auc), lw=2)
# plt 1 point in bottom left, 1 in top left, 1 in top right = perfect
plt.plot([0, 0, 1],
	    [0, 1, 1],
	    lw=2,
	    linestyle=':',
	    color='black',
	    label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curves')
plt.legend(loc='lower right')
#plot.show()
# %%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

pipe_svc = pipe_svc.fit(X_train2, y_train)
y_pred2 = pipe_svc.predict(X_test[:, [4, 14]])

roc_auc_score(y_true=y_test, y_score=y_pred2)  # param names:

accuracy_score(y_true=y_test, y_pred=y_pred2)  # notice slight difference


# %% micro average scorer option

pre_scorer = make_scorer(score_func=precision_score,
					pos_label=1,
					greater_is_better=True,
					average='micro')

# %% ==== Experiments in stacking np arrays, and loading data

r_data.data.shape
r_data.target.shape
r_data.target.reshape(-1,1).shape
a = r_data.data
b = r_data.target.reshape(-1,1)
a.shape
b.shape

# Stuff that doesn't work
np.stack((a, b), axis=1)  # doesn't stack?
np.stack((a, b), axis=0)  # still doesn't stack
np.stack([a,b], axis=0)  # not the problem either
a + b  # doesn't give desired output

# stuff that works
c = np.hstack((a,b))  # this works
c = np.concatenate((a,b), axis=1)  # works
c = np.c_[a,b]  # works, shorthand for column stack. also a r_[] function
c = np.column_stack((a,b))  # also works.


# new pipeline object - testing names and variables inside pipelines
pipe_2 = Pipeline(
		[('scl', StandardScaler()),
	     ('lrm', LogisticRegression())]
	)
%timeit  # 2nd test
train_scores_2, test_scores_2 = validation_curve(
		estimator=pipe_2,
		X=X_train,
		y=y_train,
		param_name='lrm__C',  # inside the lrm object, find the param C.
		param_range=param_range,
		cv=10,
		n_jobs=2
	)
%timeit  # 3rd test
train_scores_3, test_scores_3 = validation_curve(
		estimator=LogisticRegression(),
		X=X_train,
		y=y_train,
		param_name='C',  # inside the lrm object, find the param C.
		param_range=param_range,
		cv=10,
		n_jobs=-1
	)

