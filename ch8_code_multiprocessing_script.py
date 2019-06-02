# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 06:38:05 2018
Raschka - Python Machines Learning
Follow along

To be run as a single script (with a if __name__ == '__main__' section)
To test multi processing, which apparently doesn't work on windows without it

Used to generate (and pickle) a trained model for use

NOTE: the model used in chapter 9 is a SGD model, requires the partial_fit()
method, which a pure logistic regression does not have.

I.e. the model produced by this script isn't compatible

Assumes a saved, pre-shuffled csv movie data file which is read in first

Chapter 8 - Sentiment Analysis

@author: yehan
"""
import pandas as pd
import numpy as np
import os

# %% pre loaded/processed dataframe

df = pd.read_csv('./movieclassifier/movie_data.csv', encoding="utf8")
df.head(3)

# %% process into tokens

# easiest:
def tokenizer(text):
	return text.split()  # split on space is default anyways

# use an out of the box stemmer whilst splitting:
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]

# %% stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')


# %% train logistic regression for sentiment classification

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# from sklearn.grid_search import GridSearchCV # this one is deprecated - use the modelSelection module
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
			strip_accents=None,
			lowercase=False,
			preprocessor=None)

# different params to try in the grid search
param_grid = [
				{
			    'vect__ngram_range': [(1,1)],
				'vect__stop_words': [stop, None],
				# can even try different tokenizers
				'vect__tokenizer': [tokenizer, tokenizer_porter],
				'clf__penalty': ['l1', 'l2'],
				'clf__C': [1.0, 10.0, 100.0]
				} ,
				{
			    'vect__ngram_range': [(1,1)],
				'vect__stop_words': [stop, None],
				'vect__tokenizer': [tokenizer, tokenizer_porter],
				'vect__use_idf': [False],
				'vect__norm': [None],
				'clf__penalty': ['l1', 'l2'],
				'clf__C': [1.0, 10.0, 100.0]
			    }
		]

lr_tfidf = Pipeline([
				('vect', tfidf),
				('clf', LogisticRegression(random_state=0))
			])
# %% set up and run grid search
# see https://github.com/scikit-learn/scikit-learn/issues/5115
# parallelisation won't work wihtout a `if __name__ == '__main__'` on windows
# took 52.3 mins to train on rig r2, i7-3770 CPU

import pickle

if __name__ == '__main__':
	gs_lr_tfidf = GridSearchCV(
						lr_tfidf,
						param_grid,
						scoring='accuracy',
						cv=5,
						verbose=1,
						n_jobs=-1  # parallelisation has bugs...
						)

	gs_lr_tfidf.fit(X_train, y_train)
	print('Best parameter set:{0}'.format(gs_lr_tfidf.best_params_))
	print('CV Accuracy: {0}'.format(gs_lr_tfidf.best_score_))
	# save the best estimator
	clf = gs_lr_tfidf.best_estimator_
	print('Test Accuracy: {0}'.format(clf.score(X_test, y_test)))

	# pickling output
	dest = os.path.join('movieclassifier', 'pkl_objects')
	if not os.path.exists(dest):
		os.makedirs(dest)
	pickle.dump(stop,
		open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
		protocol=4)
	pickle.dump(clf,
		open(os.path.join(dest, 'classifier.pkl'), 'wb'),
		protocol=4)
	print('best estimator pickled as estimator.pkl')
