# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:12:22 2019
Raschka - Python Machines Learning
Follow along

Chapter 13 - Keras example
@author: Hungy
"""

# %%

""" NOTE: HOW TO RUN
run this entire script from cmd, need a theanorc config file with
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

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

os.chdir(r'E:\Dropbox\2017\Textbooks\raschka_code')
path = os.path.join(os.getcwd(), 'mnist')
# expects there to be an mnist folder in cwd with the data in it
X_train, y_train = load_mnist(path, 'train')
X_test, y_test = load_mnist(path, 't10k')

print('train:', X_train.shape, 'test:', X_test.shape)

# %% theano configs
import theano
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

from keras.utils import np_utils
print('First 3 labels:', y_train[:3])
y_train_ohe = np_utils.to_categorical(y_train)
print('\nFirst 3 labels (one-hot): \n', y_train_ohe[:3])

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

np.random.seed(1)

model = Sequential()
model.add(Dense(input_dim=X_train.shape[1],
				output_dim=50,
				init='uniform',
				activation='tanh'))

model.add(Dense(input_dim=50,
				output_dim=50,
				init='uniform',
				activation='tanh'))

model.add(Dense(input_dim=50,
				output_dim=y_train_ohe.shape[1],
				init='uniform',
				activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# %% Fit with keras
model.fit(
	X_train,
	y_train_ohe,
	nb_epoch=50,
	batch_size=300,
	verbose=1,
	validation_split=0.1,
#	show_accuracy=True
)

y_train_pred = model.predict_classes(X_train, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])
train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))