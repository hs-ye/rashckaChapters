# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 06:38:05 2018
Raschka - Python Machines Learning
Follow along

Chapter 8 - Sentiment Analysis

@author: yehan
"""
import pyprind
import pandas as pd
import numpy as np
import os

# %%
pbar = pyprind.ProgBar(50000)  # number of docs to read in, one per iteration
labels = {"pos":1, "neg":0}
df = pd.DataFrame()
a=[]

# going through the folder structure
for s in ("test", 'train'):
	for l in ('pos', 'neg'):
		#path = os.getcwd()
		# load data from the following directory (not tested)
		path = "./movieclassifier/raw_data/aclImdb/{0}/{1}".format(s, l)
		for file in os.listdir(path):
			# need to have correct encoding
			with open(os.path.join(path, file), 'r', encoding="utf8") as infile:
				a.append(os.path.join(path, file))  # for use with the test below
				txt = infile.read()
			df = df.append([ [txt, labels[l]] ], ignore_index=True)
			pbar.update() # might not work in ipython
		# check which folder it's up to
		# helps see if data is missing/not all folders are covered
		print("{0} {1} complete".format(s, l))

df.columns = ['review', 'sentiment']
# %% testing why loading files enconding doesn't work
for fpath in a[:1000]:
	with open(fpath, 'r', encoding="utf8") as b:
		c = b.read()
		print(fpath)

# have a look at what's loaded
df['sentiment'].describe

# %% reshuffle + save data to disk

import numpy as np

np.random.seed(0)
re_index = df.index
df = df.reindex(np.random.permutation(re_index))
df.to_csv('.movieclassifier/movie_data.csv', index=False, encoding="utf8")

# %% pre loaded/processed dataframe

df = pd.read_csv('.movieclassifier/movie_data.csv', encoding="utf8")
df.head(3)

# %% transforming words into feature vectors

# example only
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()  # this class can change ngram models, use ngram_range param
# CountVectorizer(ngram_range=(2,2))  # gives you 2-gram model
docs = np.array([
			'The sun is shining',
			'The weather is sweet',
			'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)

# check contents
print(count.vocabulary_)  # column and their indicies
print(bag.toarray())  # occurences of each word

# %% tf - idf

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()  # note: uses L2 normlaisation by default
np.set_printoptions(precision=2)  # print options

bag = count.fit_transform(docs)
print(tfidf.fit_transform(bag).toarray())
# can see that words that are more 'unique' to a document are weighted higher
# words used in other documents are less important

# %% cleaning text data
df.loc[0, 'review'][-50:]  # problems with the text

import re
def preprocessor(text):
	text = re.sub('<[^>]*>', '', text)  # removes html markup
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	#text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	text = re.sub('[\W]+', ' ', text.lower()) + \
		' '.join(emoticons).replace('-', '')
	return text

preprocessor(df.loc[0, 'review'][-50:])
testtext = '</a>This :) is :( a test :-)!'
preprocessor(testtext)

# run it on all the texts
df['review'] = df['review'].apply(preprocessor)

# %% process into tokens

# easiest:
def tokenizer(text):
	return text.split()  # split on space is default anyways

# use an out of the box stemmer whilst splitting:
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]


tokenizer_porter('runners like running thus they run')

# %% stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
print(stop)

[w for w in tokenizer_porter('runners like running and thus runs a lot') if w not in stop]



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
# tfidf vectorizer combines both count vectorizer and tf-idf transformer

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
# parallelisation won't work wihtout a `if __name__ == '__main__'`?
# see https://github.com/scikit-learn/scikit-learn/issues/5115
# run separate self contained script
gs_lr_tfidf = GridSearchCV(
						lr_tfidf,
						param_grid,
						scoring='accuracy',
						cv=5,
						verbose=1
						# , n_jobs=2
					  )

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set:{0}'.format(gs_lr_tfidf.best_params_))
print('CV Accuracy: {0}'.format(gs_lr_tfidf.best_score_))
# save the best estimator
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: {0}'.format(clf.score(X_test, y_test)))

# pickling output
import pickle

dest = os.path.join('movieclassifier', 'pkl_objects_v2rig')
if not os.path.exists(dest):
	os.makedirs(dest)
pickle.dump(stop,
		  open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf,
	open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)

# %% Bigger data - Online learning

import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
	text = re.sub('<[^>]*>', '', text)  # removes html markup
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
	#text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	text = re.sub('[\W]+', ' ', text.lower()) + \
		' '.join(emoticons).replace('-', '')
	tokenized = [w for w in text.split() if w not in stop]
	return tokenized

# streamer to read and return one doc at a time
def stream_docs(path):
	with open(path, 'r', encoding="utf8") as csv:
		next(csv) # skip header
		for line in csv:
			text, label = line[:-3], int(line[-2])
			# yield is the iterator equivalent of return (i think?)
			yield text, label

# test the streamer works
next(stream_docs(path='./movieclassifier/movie_data.csv'))
# %% minibatch: get a batch of the docs

def get_minibatch(doc_stream, size):
	docs, y = [], []
	try:
		for _ in range(size):
			text, label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except StopIteration:
		return None, None
	return docs, y

# %%  different vectoriser implementation, because tfidf uses the full set/not suitable for online

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
# stochastic gradient descent??

vect = HashingVectorizer(decode_error='ignore',
					n_features=2**21,  # a lot of features...
					preprocessor=None,
					tokenizer=tokenizer)
clf = SGDClassifier(loss="log", random_state=1, max_iter=1) #n_iter deprecated
doc_stream = stream_docs(path='./movieclassifier/movie_data.csv')

# %% doing the batch learning

import pyprind
pbar = pyprind.ProgBar(45)  # 45k docs, with 5k leftover for testing
classes = np.array([0, 1])  # either good or bad
for _ in range(45):
	X_train, y_train = get_minibatch(doc_stream, size=1000)
	if not X_train:
		break
	X_train = vect.transform(X_train)
	clf.partial_fit(X_train, y_train, classes=classes)
	pbar.update()
	# much faster, even on a single core

# testing using the last 5k data
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test_tfm = vect.transform(X_test)
print('Accuracy: {0}'.format(clf.score(X_test, y_test)))  # 0.878

# %% Because of partial fit/online training, can even use this data to update model again
clf = clf.partial_fit(X_test_tfm, y_test)
clf.score(X_test_tfm, y_test)  # 0.882 after update

pickle.dump(stop,
	open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
	protocol=4)
pickle.dump(clf,
	open(os.path.join(dest, 'classifier.pkl'), 'wb'),
	protocol=4)

# %% test partial fit input shape
X_test, y_test = get_minibatch(doc_stream, size=1)
print(X_test)
vect.transform(X_test)
vect.transform('this is a test review please ignore')
type(X_test)
type('this is a test review please ignore')
