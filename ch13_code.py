# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:20:00 2019
Raschka - Python Machines Learning
Follow along

Chapter 13 - Parallelizing NN Training with Theano
@author: Hungy
"""

import theano
import numpy as np
from theano import tensor as T
import matplotlib.pyplot as plt
theano.config.floatX = 'float32'

# theano programs have 3 basic steps
# initialise
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# compile
net_input = theano.function(inputs = [w1, x1, w0], outputs=z1)

#execute
print('Net input:{}'.format(net_input(2.0, 1.0, 0.5)))


# %% Configuring 32 bit for GPU

print(theano.config.floatX)
theano.config.floatX = 'float32'

# this should be run in command line?
# export THEANO_FLAGS=floatX=float32

print(theano.config.device)  # defaults to CPU
''' to run python script in command line with settings, use
THEANO_FLAGS=devie=gpu, floatx=float32 python script.py
'''

# %% Theano arrays
import numpy as np
# initialise
x = T.matrix(name='x')
x_sum = T.sum(x, axis=0)

# compile
calc_sum = theano.function(inputs=[x], outputs=x_sum)

# execute (python list)
ary = [[1, 2, 3], [1, 3, 5]]
print('Col sum', calc_sum(ary))

# execute (np array)
ary = np.array([[1, 2, 3], [2, 5 , 6]], dtype=theano.config.floatX)
print('Col sum np', calc_sum(ary))

print(x)  # we have given x a name in the initialisation
print(x.type)

# %% memory management demo
# initialize
x = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[x], updates=update, outputs=z)

# execute
data = np.array([[1, 2, 3]],  dtype=theano.config.floatX)
for i in range(5):
	print('z%d:' % i, net_input(data))

# %% keep/update data in GPU memory demo

# initialize
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
x = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
# the 'givesns' argument is what keeps data in GPU memory
net_input = theano.function(inputs=[], updates=update, givens={x: data},outputs=z)

# execute
for i in range(5):
	print('z:', net_input())

# %% OLS theano demo
# data
X_train = np.asarray([[0.0], [1.0],
					[2.0], [3.0],
					[4.0], [5.0],
					[6.0], [7.0],
					[8.0], [9.0]],
					dtype=theano.config.floatX)
y_train = np.asarray([1.0, 1.3,
					3.1, 2.0,
					5.0, 6.3,
					6.6, 7.4,
					8.0, 9.0],
					dtype=theano.config.floatX)

import theano
from theano import tensor as T
import numpy as np

def train_linreg(X_train, y_train, eta, epochs):
	costs = []

	# Initialize arrays
	eta0 = T.fscalar('eta0')
	y = T.fvector(name='y')
	X = T.fmatrix(name='X')
	w = theano.shared(np.zeros(	shape=(X_train.shape[1] + 1),
		dtype=theano.config.floatX), name='w')

	# calculate cost
	net_input = T.dot(X, w[1:]) + w[0]
	errors = y - net_input
	cost = T.sum(T.pow(errors, 2))

	# perform gradient update
	gradient = T.grad(cost, wrt=w)
	update = [(w, w - eta0 * gradient)]

	# compile model
	train = theano.function(
		inputs=[eta0],
		outputs=cost,
		updates=update,
		givens={X: X_train,
		y: y_train,}
	)

	for _ in range(epochs):
		costs.append(train(eta))

	return costs, w

# %% run and plot demo
import matplotlib.pyplot as plt
costs, w = train_linreg(X_train, y_train, eta=0.001, epochs=10)
plt.plot(range(1, len(costs)+1), costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()


# %% Predict function
def predict_linreg(X, w):
	Xt = T.matrix(name='X')
	net_input = T.dot(Xt, w[1:]) + w[0]
	predict = theano.function(inputs=[Xt],
	givens={w: w},
	outputs=net_input)
	return predict(X)

plt.figure()
plt.scatter(X_train, y_train, marker='s', s=50)
plt.plot(range(X_train.shape[0]), predict_linreg(X_train, w),
	color='gray', marker='o', markersize=4, linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


''' The parts on activation functions in the book have been skipped'''

# %%
import os
import struct
import numpy


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

path = os.path.join(os.getcwd(), 'mnist')
# expects there to be an mnist folder in cwd with the data in it
X_train, y_train = load_mnist(path, 'train')
X_test, y_test = load_mnist(path, 't10k')

X_train.shape
X_test.shape
