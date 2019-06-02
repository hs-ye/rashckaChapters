# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 21:01:51 2018
Raschka - Python Machines Learning
Follow along

Chapter 12 - Artificial Neural Networks for Image recognition
@author: yehan
"""

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`
	# Testing:
	path = os.path.join(os.getcwd(), 'mnist')
	kind = 'train'
	"""
	labels_path = os.path.join(path, '{0}-labels.idx1-ubyte'.format(kind))
	images_path = os.path.join(path, '{0}-images.idx3-ubyte'.format(kind))

	with open(labels_path, 'rb') as lbpath:
		# some magic, reading in the raw byes from the fileimgpath.read(16)
		# the magic is needed to remove the first 8 bytes, which are not data
		# but used to describe the data to file processors
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		# some more magic
		magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(
			len(labels), 784)  # hardcoded to 784 columns??

	return images, labels

# %% more tests
#with open(images_path, 'rb') as imgpath:
#	testa = imgpath.read(16)

# %% using the load function

path = os.path.join(os.getcwd(), 'mnist')
kind = 'train'
X_train, y_train = load_mnist(path, 'train')
X_test, y_test = load_mnist(path, 't10k')

X_train.shape
X_test.shape

# %% Plot one row of data

import matplotlib.pyplot as plt

# make the boxes/blank chart canvas, share X/Y means scales are the same
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()  # returns a 1d array, not sure what this means

# make the data from line into a square (ravel it)
for i in range(10):
	# the value in the middle of the [] is the number to change for different
	img = X_train[y_train == i][1].reshape(28, 28)
	ax[i].imshow(img, cmap='Greys', interpolation='nearest')

# formatting output
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# %% plot many examples of one of the numbers

fig2, ax2 = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax2 = ax2.flatten()

X_train.shape  # total
X_train[y_train == 2].shape  # for a specific value

for i in range(25):
	# the value in the middle of the [] is the number to change for different
	img = X_train[y_train == 4][i].reshape(28, 28)
	ax2[i].imshow(img, cmap='Greys', interpolation='nearest')

ax2[0].set_xticks([])
ax2[0].set_yticks([])
plt.tight_layout()
#plt.show()

# %% plot the first 25

fig2, ax2 = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax2 = ax2.flatten()

for i in range(25):
	# the value in the middle of the [] is the number to change for different
	img = X_train[:25,:][i].reshape(28, 28)
	ax2[i].imshow(img, cmap='Greys', interpolation='nearest')

ax2[0].set_xticks([])
ax2[0].set_yticks([])
plt.tight_layout()
#plt.show()

# %% Optional: Save MNIST as csv
'''
np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')

# Load from csv
X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')
X_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')
'''

# %%  set up for multi layer perceptron class

import numpy as np
from scipy.special import expit  # logistic function
import sys

class NeuralNetMLP(object):
	def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0,
		    epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0,
		    shuffle=True, minibatches=1, random_state=None):
		np.random.seed(random_state)
		self.n_output = n_output
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.w1, self.w2 = self._initialize_weights()
		self.l1 = l1
		self.l2 = l2
		self.epochs = epochs
		self.eta = eta
		self.alpha = alpha
		self.decrease_const = decrease_const
		self.shuffle = shuffle
		self.minibatches = minibatches

	def _encode_labels(self, y, k):
		'''y needs to be 1D numpy array, n is # classes
		returns len(y) * k matrix
		'''
		onehot = np.zeros((k, y.shape[0]))
		for idx, val in enumerate(y):
			onehot[val, idx] = 1.0
		return onehot

	def _initialize_weights(self):
		'''Randomnly initialise, according to #weight params needed for each layer
		w1: h x (m + 1), extra is from bias unit
		w2: t x (n + 1), extra from bias unit

		h: #hidden units in hidden layer
		m: #columns/features in input data
		t: #output classes
		n: #observations in input data

		'''
		w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
		w1 = w1.reshape(self.n_hidden, self.n_features + 1)
		w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
		w2 = w2.reshape(self.n_output, self.n_hidden + 1)
		return w1, w2

	def _sigmoid(self, z):
		# expit is equivalent to 1.0/(1.0 + np.exp(-z))
		return expit(z)

	def _sigmoid_gradient(self, z):
		"""this is the derivative of sigmoid or something"""
		sg = self._sigmoid(z)
		return sg * (1 - sg)

	def _add_bias_unit(self, X , how="column"):
		"""Adds an extra row or column of ones to the data matrix X for bias"""
		if how == 'column':
			X_new = np.ones((X.shape[0], X.shape[1] + 1))
			X_new[:, 1:] = X
		elif how == 'row':
			X_new = np.ones((X.shape[0] + 1, X.shape[1]))
			X_new[1:, :] = X
		else:
			raise AttributeError('`how` must be `column` or `row`')
		return X_new

	def _feedforward(self, X, w1, w2):
		"""
		A main function

		Carry forward all matrix calculations through the neural net using given weights
		w1: weights for layer 1,
		w2: weights for layer 2
		"""
		a1 = self._add_bias_unit(X, how="column")
		z2 = w1.dot(a1.T)  # transpose here
		a2 = self._sigmoid(z2)
		# print("a2 dim: ", a2.shape)  # debugging use
		a2 = self._add_bias_unit(a2, how="row")
		z3 = w2.dot(a2)  # don't need transpose here from algorithm design
		a3 = self._sigmoid(z3)
		return a1, z2, a2, z3, a3

	def _L2_reg(self, lambda_, w1, w2):
		'''calculating some scalar parameter, assume for regularisation (squares)
		the w1[:, 1:] part is probably removing the bias unit?'''
		return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

	def _L1_reg(self, lambda_, w1, w2):
		'''calculating some scalar parameter, assume for regularisation (abs)
		the w1[:, 1:] part is probably removing the bias unit?'''
		return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

	def _get_cost(self, y_enc, output, w1, w2):
		"""
		A main function

		Overall cost function of the fitted perceptron, vs the one hot encoded y labels
		y_enc is pre-encoded y label matrix"""
		term1 = -y_enc * (np.log(output))
		term2 = (1 - y_enc) * np.log(1- output)
		cost = np.sum(term1 - term2)
		L1_term = self._L1_reg(self.l1, w1, w2)
		L2_term = self._L2_reg(self.l2, w1, w2)
		cost = cost + L1_term + L2_term
		return cost

	def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
		'''
		A main function

		# backpropagation, returns the gradients required to update weights
		# after calculating the errors and propagating it backwards through layers
		'''
		sigma3 = a3 - y_enc
		z2 = self._add_bias_unit(z2, how="row")
		sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
		sigma2 = sigma2[1:, :]
		grad1 = sigma2.dot(a1)
		grad2 = sigma3.dot(a2.T)

		# regularise (modify gradient, except the bias unit)
		grad1[:, 1:] += self.l2 * w1[:, 1:]
		grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
		grad2[:, 1:] += self.l2 * w2[:, 1:]
		grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

		return grad1, grad2

	def predict(self, X):
		''' What the user runs to predict'''

		a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
		y_pred = np.argmax(z3, axis=0)
		return y_pred

	def fit(self, X, y, print_progress=False):
		''' What the user runs to fit'''
		self.cost_ = []
		X_data, y_data = X.copy(), y.copy()
		y_enc = self._encode_labels(y, self.n_output)
		# setting up tracking of previous weights for layer1/2
		delta_w1_prev = np.zeros(self.w1.shape)
		delta_w2_prev = np.zeros(self.w2.shape)

		for i in range(self.epochs):
			# learning each epoch
			# adaptive learning rate
			self.eta /= (1 + self.decrease_const * i)

			if print_progress:
				sys.stderr.write(
					'\rEpoch: {0}/{1}'.format(i+1, self.epochs))
				sys.stderr.flush()

			if self.shuffle:
				idx = np.random.permutation(y_data.shape[0])
				X_data, y_enc = X_data[idx], y_enc[:,idx]

			# start the learning within each epoch in batches
			mini = np.array_split(range(y_data.shape[0]), self.minibatches)
			# print(mini)  # debugging - splits the indicies of obs using length of y
			# splits into batches, each batch is an np array of row ids
			for idx in mini: # idx is an np array of row ids
				# feedforward using current weights
				a1, z2, a2, z3, a3 = self._feedforward(X_data[idx], self.w1, self.w2)
				# only compare cost 1 label at a time
				cost = self._get_cost(y_enc=y_enc[:,idx], output=a3, w1=self.w1, w2=self.w2)
				# then
				self.cost_.append(cost)

				# compute gradient via back propagation
				grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3,
					z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)

				# update weights, which is attached to the model instance itself
				delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
				self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
				self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
				delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

		# model updates is recorded in the instance
		return self




# %% set up and fit

nn = NeuralNetMLP(
		n_output=10,
		n_features=X_train.shape[1],
		n_hidden=50,
		l1=0.0,
		l2=0.1,
		epochs=200,  # testing - shuold be enough
		# epochs=1000,  # this will take 15-30 mins on an i5 laptop
		eta=0.001,
		alpha=0.001,
		decrease_const=0.00001,
		shuffle=True,
		minibatches=50,
		random_state=1
	)
nn.fit(X_train, y_train, True)

# %% check results against training

print(max(nn.cost_), " ", min(nn.cost_))
plt.plot(range(len(nn.cost_)), nn.cost_)  # plot cost decrease per epoch
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
acc  # if acc too low (e.g. only 6%) Something's wrong
# see errors below

# 200 Epochs: 93.4%

# %% start looking for problems, if any
y_train[0:100]  # most predictions are 1's...definitely not right
y_train_pred[0:100]  # most predictions are 1's...definitely not right
"""
Symptom: most predictions are 1's
most likely issue: y not encoded properly, if it's only 1 or 0 or using only
whether it's a 1 or not, then model will be incorrectly trained
"""

# %% plot moving average of costs over batch, to show trend

# averaged over each epoch
n_epochs = 200
batches = np.array_split(range(len(nn.cost_)), n_epochs)  # 1000 arrays of indicies
cost_ary = np.array(nn.cost_)  # gets the cost in a numpy array
# use each batch as index, calculate avreages, get np array of 1000 cost avgs
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(n_epochs), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.tight_layout()
plt.show()

# %% check results against test
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test_pred == y_test, axis=0) / X_test.shape[0]
print("test acc: {}%".format(acc))
# 200 epochs: 93.13%

# %% look for images which are not correctly predicted
bad_img = y_test_pred != y_test  # store the index
miscl_img = X_test[bad_img][:25]  # use index on array, first 25
correct_lab = y_test[bad_img][:25]
miscl_lab = y_test_pred[bad_img][:25]

fig ax = plt.subplots(nrows=5
					  ncols=5,
					  sharex=True,
					  sharey=True)

ax = ax.flatten()
for i in range(25):
	img=miscl_img[i].reshape(28,28)
	ax[i].imshow(img, cmap='Greys', interpolation='nearest')
	ax[i].set_title('{0}) t: {1} p: {2}'.format(
			i+1, correct_lab[i], miscl_lab[i]
		)
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# %% New NN class with gradient checking
class MLPGradientCheck(object):
	def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0,
		    epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0,
		    shuffle=True, minibatches=1, random_state=None):
		np.random.seed(random_state)
		self.n_output = n_output
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.w1, self.w2 = self._initialize_weights()
		self.l1 = l1
		self.l2 = l2
		self.epochs = epochs
		self.eta = eta
		self.alpha = alpha
		self.decrease_const = decrease_const
		self.shuffle = shuffle
		self.minibatches = minibatches

		self.check_num_grads = []
		self.check_grads = []

	def _encode_labels(self, y, k):
		'''y needs to be 1D numpy array, n is # classes
		returns len(y) * k matrix
		'''
		onehot = np.zeros((k, y.shape[0]))
		for idx, val in enumerate(y):
			onehot[val, idx] = 1.0
		return onehot

	def _initialize_weights(self):
		'''Randomnly initialise, according to #weight params needed for each layer
		w1: h x (m + 1), extra is from bias unit
		w2: t x (n + 1), extra from bias unit

		h: #hidden units in hidden layer
		m: #columns/features in input data
		t: #output classes
		n: #observations in input data

		'''
		w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
		w1 = w1.reshape(self.n_hidden, self.n_features + 1)
		w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
		w2 = w2.reshape(self.n_output, self.n_hidden + 1)
		return w1, w2

	def _sigmoid(self, z):
		# expit is equivalent to 1.0/(1.0 + np.exp(-z))
		return expit(z)

	def _sigmoid_gradient(self, z):
		"""this is the derivative of sigmoid or something"""
		sg = self._sigmoid(z)
		return sg * (1 - sg)

	def _add_bias_unit(self, X , how="column"):
		"""Adds an extra row or column of ones to the data matrix X for bias"""
		if how == 'column':
			X_new = np.ones((X.shape[0], X.shape[1] + 1))
			X_new[:, 1:] = X
		elif how == 'row':
			X_new = np.ones((X.shape[0] + 1, X.shape[1]))
			X_new[1:, :] = X
		else:
			raise AttributeError('`how` must be `column` or `row`')
		return X_new

	def _feedforward(self, X, w1, w2):
		"""
		A main function

		Carry forward all matrix calculations through the neural net using given weights
		w1: weights for layer 1,
		w2: weights for layer 2
		"""
		a1 = self._add_bias_unit(X, how="column")
		z2 = w1.dot(a1.T)  # transpose here
		a2 = self._sigmoid(z2)
		# print("a2 dim: ", a2.shape)  # debugging use
		a2 = self._add_bias_unit(a2, how="row")
		z3 = w2.dot(a2)  # don't need transpose here from algorithm design
		a3 = self._sigmoid(z3)
		return a1, z2, a2, z3, a3

	def _L2_reg(self, lambda_, w1, w2):
		'''calculating some scalar parameter, assume for regularisation (squares)
		the w1[:, 1:] part is probably removing the bias unit?'''
		return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

	def _L1_reg(self, lambda_, w1, w2):
		'''calculating some scalar parameter, assume for regularisation (abs)
		the w1[:, 1:] part is probably removing the bias unit?'''
		return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

	def _get_cost(self, y_enc, output, w1, w2):
		"""
		A main function

		Overall cost function of the fitted perceptron, vs the one hot encoded y labels
		y_enc is pre-encoded y label matrix"""
		term1 = -y_enc * (np.log(output))
		term2 = (1 - y_enc) * np.log(1- output)
		cost = np.sum(term1 - term2)
		L1_term = self._L1_reg(self.l1, w1, w2)
		L2_term = self._L2_reg(self.l2, w1, w2)
		cost = cost + L1_term + L2_term
		return cost

	def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
		'''
		A main function

		# backpropagation, returns the gradients required to update weights
		# after calculating the errors and propagating it backwards through layers
		'''
		sigma3 = a3 - y_enc
		z2 = self._add_bias_unit(z2, how="row")
		sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
		sigma2 = sigma2[1:, :]
		grad1 = sigma2.dot(a1)
		grad2 = sigma3.dot(a2.T)

		# regularise (modify gradient, except the bias unit)
		grad1[:, 1:] += self.l2 * w1[:, 1:]
		grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
		grad2[:, 1:] += self.l2 * w2[:, 1:]
		grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

		return grad1, grad2

	def predict(self, X):
		''' What the user runs to predict'''

		a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
		y_pred = np.argmax(z3, axis=0)
		return y_pred

	def _gradient_checking(self, X, y_enc, w1, w2, epsilon, grad1, grad2):
		""" apply gradient checking for debugging
		Works for this single hidden layer NN only, not generalised
		Returns
		---------
		relative error: float
		relative numerical error from the estimated gradients, vs the
		model produced backprop gradients

		Want this to be small: rules of thumb:
			re < 1e-7: should be fine
			re > 1e-4: problems in model
		"""
		# layer 1
		num_grad1 = np.zeros(w1.shape)
		epsilon_ary1 = np.zeros(w1.shape)
		for i in range(w1.shape[0]):
			for j in range(w1.shape[1]):
				# to calculate the partial derivative for this gradient only
				# set only this one with a non-zero eps leave all other as 0
				epsilon_ary1[i,j] = epsilon
				# add and subtract eps from the relevant W value
				a1, z2, a2, z3, a3 = self._feedforward(X, w1 - epsilon_ary1, w2)
				# then get the cost
				cost1 = self._get_cost(y_enc, a3, w1 - epsilon_ary1, w2)
				a1, z2, a2, z3, a3 = self._feedforward(X, w1 + epsilon_ary1, w2)
				cost2 = self._get_cost(y_enc, a3, w1 + epsilon_ary1, w2)
				# calculate numeric grad, then reset the eps
				num_grad1[i, j] = (cost2 - cost1) / (2 * epsilon)
				epsilon_ary1[i, j] = 0

		num_grad2 = np.zeros(np.shape(w2))
		epsilon_ary2 = np.zeros(np.shape(w2))
		for i in range(w2.shape[0]):
			for j in range(w2.shape[1]):
				# to calculate the partial derivative for this gradient only
				# set only this one with a non-zero eps leave all other as 0
				epsilon_ary2[i,j] = epsilon
				# add and subtract eps from the relevant W value
				a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 - epsilon_ary2)
				# then get the cost
				cost1 = self._get_cost(y_enc, a3, w1 , w2 - epsilon_ary2)
				a1, z2, a2, z3, a3 = self._feedforward(X, w1 , w2 + epsilon_ary2)
				cost2 = self._get_cost(y_enc, a3, w1 , w2 + epsilon_ary2)
				# calculate numeric grad, then reset the eps
				num_grad2[i, j] = (cost2 - cost1) / (2 * epsilon)
				epsilon_ary2[i, j] = 0

		num_grad = np.hstack((num_grad1.flatten(), num_grad2.flatten()))
		grad = np.hstack((grad1.flatten(), grad2.flatten()))
		norm1 = np.linalg.norm(num_grad - grad)
		norm2 = np.linalg.norm(num_grad)
		norm3 = np.linalg.norm(grad)
		relative_err = norm1 / (norm2 + norm3)
#		print('norm1:{0}, \nnorm2: {1}, \nnorm3:{2}'.format(
#				norm1, norm2, norm3))
		print(num_grad)
		print(grad)
		self.check_num_grads.append(num_grad)
		self.check_num_grads.append(grad)
		return relative_err

	def fit(self, X, y, print_progress=False):
		''' What the user runs to fit'''
		self.cost_ = []
		X_data, y_data = X.copy(), y.copy()
		y_enc = self._encode_labels(y, self.n_output)
		# setting up tracking of previous weights for layer1/2
		delta_w1_prev = np.zeros(self.w1.shape)
		delta_w2_prev = np.zeros(self.w2.shape)

		for i in range(self.epochs):
			# learning each epoch
			# adaptive learning rate
			self.eta /= (1 + self.decrease_const * i)

			if print_progress:
				sys.stderr.write(
					'\rEpoch: {0}/{1}'.format(i+1, self.epochs))
				sys.stderr.flush()

			if self.shuffle:
				idx = np.random.permutation(y_data.shape[0])
				X_data, y_enc = X_data[idx], y_enc[:,idx]

			# start the learning within each epoch in batches
			mini = np.array_split(range(y_data.shape[0]), self.minibatches)
			# print(mini)  # debugging - splits the indicies of obs using length of y
			# splits into batches, each batch is an np array of row ids
			for idx in mini: # idx is an np array of row ids
				# feedforward using current weights
				a1, z2, a2, z3, a3 = self._feedforward(X_data[idx], self.w1, self.w2)
				# only compare cost 1 label at a time
				cost = self._get_cost(y_enc=y_enc[:,idx], output=a3, w1=self.w1, w2=self.w2)
				# then
				self.cost_.append(cost)

				# compute gradient via back propagation
				grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3,
					z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)

				# gradient checking
				grad_diff = self._gradient_checking(
						X=X[idx],
						y_enc=y_enc[:, idx],
						w1=self.w1,
						w2=self.w2,
						epsilon=1e-5,
						grad1=grad1,
						grad2=grad2
					)
				if grad_diff <= 1e-7:
					print('Gradient OK: {}'.format(grad_diff))
				elif grad_diff <= 1e-4:
					print('Warnings: {}'.format(grad_diff))
				else:
					print('Gradient ERROR:{}'.format(grad_diff))


				# update weights, which is attached to the model instance itself
				delta_w1 = self.eta * grad1
				delta_w2 = self.eta * grad2
				self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
				self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
				delta_w1_prev = delta_w1
				delta_w2_prev = delta_w2

		# model updates is recorded in the instance
		return self
# %% run the model
# some unsolved issue here - gradient error is too large, even though
# the model itself seems to converge fine?
nn_check = MLPGradientCheck(
		n_output=10,
		n_features=X_train.shape[1],
		n_hidden=10,
		l2=0.0,
		l1=0.0,
		epochs=10,
		eta=0.001,
		alpha=0.0,
		decrease_const=0.0,
		minibatches=1,
		random_state=1)

# very expensive, so run it with 5 samples only
nn_check.fit(X_train[:5], y_train[:5], print_progress=False)