# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:25:39 2017

Raschka - Python Machines Learning
Follow along

Chapter 5 - Dimensionality Reduction
@author: yehan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% == Part 1 PCA by hand
from sklearn.datasets import load_wine
# load data - get columns in same order so that y is first

# Internet source:
#df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
#				  header=None)

# Built in from Sklearn library
	#df_wine = pd.DataFrame(load_wine().data)
	#df_wine.columns = load_wine().feature_names
	#df_wine['Class labels'] = load_wine().target
	#cols = df_wine.columns.tolist()  # gets list of column names
	#cols = cols[-1:] + cols[:-1]   # moves last to first
	#df_wine = df_wine[cols]  # re-makes the data so that order is different

# Or, could have concat the data in the correct order...
df_wine = pd.concat([pd.DataFrame(load_wine().target), pd.DataFrame(load_wine().data)],
					 axis=1)
df_wine.columns = ['Class labels'] + load_wine().feature_names

# %% Split to train/test & standardise

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test,  y_train, y_test = \
					train_test_split(X, y, test_size=0.3, random_state=0)
tfn_std = StandardScaler()
X_train_std = tfn_std.fit_transform(X_train)
X_test_std = tfn_std.fit_transform(X_test)

# %% Construct the convariance matrix (also correlation due to standardisation)
# then eigen decomposition
cov_mat = np.cov(X_train_std.T)  # not sure why the transpose is required?
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vals)
print(eigen_vecs)

# find % explained by size of eigen values (not yet linked to their eigenvectors)
tot = sum(eigen_vals)  # sum of eigenvalues
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]  # relative eigenvalue size
cum_var_exp = np.cumsum(var_exp)

# %% plot

import matplotlib.pyplot as plt

plt.figure(1)
plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual expl. var.')
plt.step(range(1,14), cum_var_exp, where='mid', label='cum. variance ratio')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
#plt.show()


# %% now get the eigenvecs corresponding to highest eigen vals
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)  # Note: sorting works on element by element of a tuple
# Skips things it doesn't know

# projection marix from top 2 eigen vectors
#w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))  # old
w = np.stack((eigen_pairs[0][1], eigen_pairs[1][1]), axis=1)  # preferred not to use hstack


# %% Use projection to reduce dimensions
# test projection of 1 observation
X_train_std[0].dot(w)

# projection of entire data
X_train_pca = X_train_std.dot(w)

# and plot it in 2d space, with classes
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
plt.figure(2)
for l, c, m in zip(list(set(y_train)), colors, markers):
	plt.scatter(X_train_pca[y_train==l,0], X_train_pca[y_train==l,1],
			    c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
# plt.show()

# %% == Part 2 PCA with sklearn

# %% Make a chart V2, from the chapter 3 code, slightly improved
from matplotlib.colors import ListedColormap
#from ch3_code import plot_decision_regions  # this won't work due to scripts not in main()

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02,
					 xlab='X', ylab='Y', legend='best', fig=0):
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
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.legend(loc=legend)

# %%  draw the decision regions using simple logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
mod_lrm = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
mod_lrm.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=mod_lrm, xlab='PC 1',
				  ylab='PC 2', fig=3)
# notice plot is mirror image - explained in Raschka. Sign Doesn't matter for prediction

# %% check on test data

plot_decision_regions(X_test_pca, y_test, classifier=mod_lrm, xlab='PC 1',
				  ylab='PC 2', fig=4)

# %% == Part 2 - LDA
# Uses same standardised data as PCA, up to X_train_std

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(3):
	mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
	print('MV {}: {}\n'.format(label, mean_vecs[label]))

# %%
# Within scatter matricies, S_W
d = 13  # num features
S_W = np.zeros((d, d))  # d x d zeros
for label, mv in zip(range(3), mean_vecs):  # mv is mean
	class_scatter = np.zeros((d, d))
	for row in X[y == label]:
		row, mv = row.reshape(d,1), mv.reshape(d, 1)  # get mv and row reshaped
		class_scatter += (row-mv).dot((row-mv).T)  # calculates x.T(x) = x^2, x is difference

	S_W += class_scatter
print('Within Class scatter matrix: {0}x{1}'.format(S_W.shape[0], S_W.shape[1]))

# print class label distributions
np.bincount(y_train)  # see they are not uniformly distributed
# this means need to re-do scatter matrix calc, scale them (so it becomes a co-var matrix)

# %% Scaled scatter matrix
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(3), mean_vecs):  # label is y's
	class_scatter = np.cov(X_train_std[y_train==label].T)  # covar for X features, within each y
	S_W += class_scatter  # add the result to the scatter matrix
print('Scaled within-class scatter: {0}x{1}'.format(S_W.shape[0], S_W.shape[1]))

# %% Between class mean
mean_overall = np.mean(X_train_std, axis=0)  # find overall mean
d = 13
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):  # for each mean vec
	n = X[y==i, :].shape[0]  # find the number of classes
	mean_vec = mean_vec.reshape(d, 1)  # make it 1 column
	mean_overall = mean_overall.reshape(d, 1)
	S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: {0}x{1}'.format(S_B.shape[0], S_B.shape[1]))  # checking


# %% Use Eigen decomp to get the LDAs done
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))  # 1 line decomposition

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for
				i in range(len(eigen_vals))]
# use lambda to get the first element of the eigenpair, to perform the sort on
eigen_pairs.sort(key=lambda k: k[0], reverse=True)  # .sort() does an in place sort
for eig_stuff in eigen_pairs:
	print(eig_stuff[0])  # only first 2 eigs are non zero
	# This is odd - eigenvalues are different from book, but eigenvectors are the same
	# when used to generate the transformation matrix (further down)

# %% plot the cumulative eigen stuff
tot = sum(eigen_vals.real)  # wow it has nonreal parts
#discr = [(i / tot) for i in eigen_vals.real.sort(reverse=True)]  # this won't work
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.figure(5)
plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='indivi. discriminability')
plt.step(range(1, 14), cum_discr, where='mid', label='cum. discriminability')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
#plt.show()

# %% Stack vectors to make transformation matrix, then apply transform
w = np.stack((eigen_pairs[0][1].real, eigen_pairs[1][1].real), axis=1)
X_train_lda = X_train_std.dot(w)  # matrix product to get transformed
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
plt.figure(6)
for l, c, m in zip(list(set(y_train)), colors, markers):
	plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1],
			   c=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
#plt.show()  # dones

# %% LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

tfm_lda = LDA(n_components=2)
X_train_lda = tfm_lda.fit_transform(X_train_std, y_train)  # needs y, supervised
# logistic regression
mod_lrm = LogisticRegression()
mod_lrm = mod_lrm.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=mod_lrm, fig=7,
				  xlab='LD 1', ylab='LD 2', legend='best')

#plt.show()

# %% test set
X_test_lda = tfm_lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=mod_lrm, fig=8,
				  xlab='LD 1', ylab='LD 2', legend='best')


# %% Implementing Kernel PCA

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh  # calculated eigenstuffs
#from numpy.linalg import eigh  # a 'lite' implementation of the scipy ver.
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
	"""
	RBF kernel PCA implementation.

	Parameters
	------------
	X: {NumPy ndarray}, shape = [n_samples, n_features]

	gamma: float
	Tuning parameter of the RBF kernel

	n_components: int
	Number of principal components to return

	Returns
	------------
	X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
	Projected dataset

	"""
	# == First part: makes the RBF Kernel calculations
	# Calculate pairwise squared euclidean distances in M x N dataset,
	# makes a list length M *N row matrix, 1 column
	sq_dists = pdist(X, 'sqeuclidean')

	# Convert pairwise distances into a square matrix
	mat_sq_dists = squareform(sq_dists)

	# Compute symmetric kernel matrix
	K = exp(-gamma * mat_sq_dists)  # there will be non-0 ones
	# maps the distances to between 1 and 0

	# center kernel matrix
	N = K.shape[0]  # should be the same
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	# == 2nd Part: Does the PCA on the kernel
	# Obtain Eigenpairs from the centered kernel matrix
	# numpy.eigh returns them in sorted order
	eigvals, eigvecs = eigh(K)  # eighvecs is square matrix

	# Collect top k eigenvectors  (projection matrix). column stack just like hstack
	# X_pc = np.column_stack((eigvecs[:,-i] for i in range(1, n_components + 1)))
	# produces m rows, with n columns depending on the n_components
	X_pc = np.stack((eigvecs[:,-i] for i in range(1, n_components + 1)), axis=1)
	# new version using the unified stack function
	return X_pc


# %% Apply rbf kernel on examples
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)
plt.figure(9)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='s', alpha=0.5)
#plt.show()

# %% Import standard PCA
from sklearn.decomposition import PCA
tfm_pca = PCA(n_components=2)  # pca transform
X_pca = tfm_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# plot 2 components first
ax[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='b', marker='s', alpha=0.5)
# now only plot 1st component
ax[1].scatter(X_pca[y==0, 0], np.zeros([50, 1]) + 0.02,  # small offset
		    color='r', marker='^', alpha=0.5)
ax[1].scatter(X_pca[y==1, 0], np.zeros([50, 1]) - 0.02, # so the spread can be seen
		    color='b', marker='s', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
#plt.show()

# %% do the Kernel PCA
from matplotlib.ticker import FormatStrFormatter

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)  # Note
# this returns a straight copy of the kpca X, not a transformer
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='b', marker='s', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros([50, 1]) + 0.02,  # small offset
		    color='r', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros([50, 1]) - 0.02, # so the spread can be seen
		    color='b', marker='s', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))  # 2 decimals
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))


# %% Separate concentric circles

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.figure(10)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b', marker='o', alpha=0.5)
#plt.show()

# %% standard PCA

tfm_pca = PCA(n_components=2)  # same as before
X_pca = tfm_pca.fit_transform(X)  # only 1 operation possible - linear PCA
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# code actually same as the half moons, just with more samples/obs generated
ax[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='b', marker='s', alpha=0.5)
ax[1].scatter(X_pca[y==0, 0], np.zeros([500, 1]) + 0.01,  # small offset
		    color='r', marker='^', alpha=0.5)
ax[1].scatter(X_pca[y==1, 0], np.zeros([500, 1]) - 0.01, # so the spread can be seen
		    color='b', marker='s', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])  # no y-ticks, not plotting any components here
ax[1].set_xlabel('PC1')

# %% kernel PCA -- same as the half moons, with more samples

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)  # Note
# this returns a straight copy of the kpca X, not a transformer
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='b', marker='s', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros([500, 1]),
		    color='r', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros([500, 1]),
		    color='b', marker='s', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))  # 2 decimals
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

# %% Modified rbf_kerne_pca so it also returns eigenvalues of kernel matrix

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh  # calculated eigenstuffs
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
	"""
	RBF kernel PCA implementation.

	Parameters
	------------
	X: {NumPy ndarray}, shape = [n_samples, n_features]

	gamma: float
	Tuning parameter of the RBF kernel

	n_components: int
	Number of principal components to return

	Returns
	------------
	X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
	Projected dataset

	lamdas: list Eigenvalues

	"""
	# == First part: makes the RBF Kernel calculations
	# Calculate pairwise squared euclidean distances in M x N dataset,
	# makes a list length M *N row matrix, 1 column
	sq_dists = pdist(X, 'sqeuclidean')

	# Convert pairwise distances into a square matrix
	mat_sq_dists = squareform(sq_dists)

	# Compute symmetric kernel matrix
	K = exp(-gamma * mat_sq_dists)  # there will be non-0 ones
	# maps the distances to between 1 and 0

	# center kernel matrix
	N = K.shape[0]  # should be the same
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	# == 2nd Part: Does the PCA on the kernel
	# Obtain Eigenpairs from the centered kernel matrix
	# numpy.eigh returns them in sorted order
	eigvals, eigvecs = eigh(K)  # eighvecs is square matrix

	# Collect top k eigenvectors  (projection matrix). column stack just like hstack
	# produces m rows, with n columns depending on the n_components
	X_pc = np.stack((eigvecs[:,-i] for i in range(1, n_components + 1)), axis=1)

	# Collect corresponding eigenvals
	lambdas = [eigvals[-i] for i in range(1,n_components+1)]
	return X_pc, lambdas

# %% remake half moon and project to 1-D subspace (1st component)
# doing a check between the formula, then multiple trans matrix by hand

X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
# projecta single point using the PCA
x_new = X[25]
x_new
x_proj = alphas[25]
x_proj  # original projection

# do a projection by hand, multiplying the trans formation matrix
def project_x(x_new, X, gamma, alphas, lambdas):
	pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
	k = np.exp(-gamma * pair_dist)  # map to 0-1 interval
	return k.dot(alphas/lambdas)  # vectors scaled by lambdas

x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj  # same as

# %% Plot out the original transformation and the hand made one
plt.figure(11)
plt.scatter(alphas[y==0, 0], np.zeros((50)),
		  color='red', marker='^',alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)),
			color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
			label='original projection of point X[25]',
			marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
			label='remapped point X[25]',
			marker='x', s=500)
plt.legend(scatterpoints=1)
#plt.show()

# %% Kernel PCA using scikitleran API

from sklearn.decomposition import KernelPCA

# Data is moon data as per before - won't repeat here
tfm_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)  # transformer obj
X_kpca = tfm_kpca.fit_transform(X)  # have to do it this way to use transformer

# plotting - see previous KPCA with the half moons - exactly the same code






