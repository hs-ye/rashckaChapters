# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:06:05 2018
Raschka - Python Machines Learning
Follow along

Chapter 9 - Embedding Machine Learning to Web Apps
@author: yehan
"""

# %% vectorizer.py
# this is to be saved as a vetorizer.py file for use as a module

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

# cur_dir = os.path.dirname(__file__)  # not sure what this is
cur_dir = os.getcwd()
stop = pickle.load(open(
		os.path.join(cur_dir,
		'movieclassifier', # where i'm putting the data, remove in vectorizer.py
		'pkl_objects',
		'stopwords.pkl'), 'rb')
		)

def tokenizer(text):
	text = re.sub('<[^>]*>', '', text)  # removes html markup
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
	#text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	text = re.sub('[\W]+', ' ', text.lower()) + \
		' '.join(emoticons).replace('-', '')
	tokenized = [w for w in text.split() if w not in stop]
	return tokenized

vect = HashingVectorizer(decode_error='ignore',
					n_features=2**21,
					preprocessor=None,
					tokenizer=tokenizer)













