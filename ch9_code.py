# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:06:05 2018
Raschka - Python Machines Learning
Follow along

Chapter 9 - Embedding Machine Learning to Web Apps
@author: yehan
"""

# serialise existing fitted model
import pickle
import os

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
	os.makedirs(dest)
# assumes you have existing stop and clf objects, see ch8 code
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
		  protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'),
		  protocol=4)


# %% extract pickled objects
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


# %% load pickled classifier object with help from the vectorizer script file
from movieclassifier.vectorizer import vect # slightly differet dir from textbook
# need to include the __init__.py blank marker file so python treats as library
import pickle
import re
import os

clf = pickle.load(open(
		os.path.join('movieclassifier','pkl_objects', 'classifier.pkl'),
		'rb'))

clf.steps[1][1]

# %% use to make prediction
import numpy as np
label = {0:'negative', 1:'positive'}

example = ['This is a great movie']
example = ['I love this movie']
example = ['This movie sucks greatly']

#X = vect.transform(example)  # not sure this step is needed, the predictor
# pipeline already has vect.transform, so this step is redundant

X = example
print('Prediction: {0} probablity: {1}'.format(
			label[clf.predict(X)[0]],
			np.max(clf.predict_proba(X)*100)  # returns the prob of all classes
			# so use max to get the relevant one
		))

clf.predict(['yes blahblah'])[0]

# %% Set up SQLite

import sqlite3
import os
conn = sqlite3.connect('./movieclassifier/reviews.sqlite')
# makes a new sqlite db file, if the file doesn't already exist
c = conn.cursor()
#c.execute("drop table if exists review_db")  # will drop and recreate table
c.execute('''CREATE TABLE review_db
			(review TEXT, sentiment INTEGER, date TEXT)''')
example1 = 'I love this movie'
c.execute("""INSERT INTO review_db
			(review, sentiment, date) VALUES
			(?, ?, DATETIME('now'))""", (example1, 1))
example2 = "This movie was bad"
c.execute("""INSERT INTO review_db
			(review, sentiment, date) VALUES
			(?, ?, DATETIME('now'))""", (example2, 1))

conn.commit()
conn.close()

# %% look at DB
conn = sqlite3.connect('./movieclassifier/reviews.sqlite')
c = conn.cursor()
c.execute("Select * from review_db")
results = c.fetchall()
conn.close()
print(results)

# or use pandas...
import pandas as pd
pd.read_sql("select * from review_db", con=conn)

# %% Updating the model permanently (serialising/deserialising)

import pickle
import sqlite3
import numpy as np
import os
# import pandas as pd  # testing
# use the local implementation of hashing vectorizer
from vectorizer import vect


def update_model(db_path, model, batch_size=10000):
	conn = sqlite3.connect(db_path)
	# conn = sqlite3.connect(db)  # testing
	# pd.read_sql("SELECT * FROM review_db", con=conn)  #testing
	c = conn.cursor()
	c.execute('SELECT * FROM review_db')

	results = c.fetchmany(batch_size)
	while results:
		# when results are not empty
		data = np.array(results)
		X = data[:, 0]
		y = data[:, 1].astype(int)
		print('{0} rows data fetched'.format(X.shape[0]))
		classes = np.array([0, 1])
		X_train = vect.transform(X)
		model.partial_fit(X_train, y, classes=classes)
		# recursive fetch so repeat this loop until done
		results = c.fetchmany(batch_size)

	conn.close()
	return None


# cur_dir = os.path.dirname(__file__)  # used in standalone script/module
cur_dir = os.getcwd() + '\\movieclassifier'
clf = pickle.load(open(os.path.join(cur_dir,
							 'pkl_objects',
							 'classifier.pkl'), 'rb'))

db = os.path.join(cur_dir, 'reviews.sqlite')

# %% do the update - get new updated model
update_model(db, clf, batch_size=10000)

pickle.dump(clf, open(os.path.join(cur_dir,
							'pkl_objects',
							'classifier_updated.pkl'), 'wb'), protocol=4)
# operational factors to consider:
# - how to keep track of what data has already been used
# - model archiving & version management
# - how to schedule update function to run regularly







