# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:06:05 2018
Raschka - Python Machines Learning
Follow along - the model updater script from database entries

Chapter 9 - Embedding Machine Learning to Web Apps
@author: yehan
"""

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