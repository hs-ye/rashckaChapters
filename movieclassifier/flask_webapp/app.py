# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:06:05 2018
Raschka - Python Machines Learning
Follow along

Chapter 9 - Embedding Machine Learning to Web Apps
@author: yehan
"""

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from sklearn.linear_model import SGDClassifier  # import in case issues
# import the HashingVectorizer instance defined in local dir script
from vectorizer import vect, tokenizer

### Prepare classifier methods
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects/classifier.pkl'), 'rb'))
print('debug: setting n iter to 1')   # debug
clf.n_iter = 1  # not having this this breaks the hosted webapp on pythonanywhere, no idea why??
db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
	'''given a document, return the label and the probabilities'''
	label = {0:'negative', 1:'positive'}
	# X = list(document)
	X = vect.transform([document])  # seems this step is not needed if clf object has tfm pipeline
	y = clf.predict(X)[0]
	proba = np.max(clf.predict_proba(X))
	return label[y], proba

def train(document, y):
	'''run the partial fit agains the current up to date clf object
	Input types:
		document should be a list
		y should be an int of 1 or 0, when it's passed in

	'''
	# X = document
	print('doc: ', document)
	print('doc tfm: ', vect.transform([document]))
	X = vect.transform([document])  # not needed if the clf object is a pipeline with processing
	clf.partial_fit(X, [y])  
	# clf.steps[1][1].partial_fit(X, y)  # use this if clf is a pipeline

def sqlite_entry(path, document, y):
	""" Open the sqlite connection, save the data, commit changes and close"""
	conn = sqlite3.connect(path)
	c = conn.cursor()
	c.execute("""INSERT INTO review_db (review, sentiment, date)
		VALUES (?, ?, DATETIME('now'))
		""", (document, y))
	conn.commit()
	conn.close()



app = Flask(__name__)

class ReviewForm(Form):
	''' assuming the validators are checks on text inputs before they are allowed to be submitted'''
	moviereview = TextAreaField('',
		[validators.DataRequired(),
		validators.length(min=15)])

@app.route('/')
def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():
		# get the data from the moviereview request sent
		review = request.form['moviereview']
		# get predicted y and prob using the classify method on the data
		y, proba = classify(review)
		# send this data to the template so it can get rendered
		return render_template(
				'results.html',
				content=review,
				prediction=y,
				probability=round(proba*100, 2)
			)
		return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
	""" Asks users for feedback after generating a prediction from their reviews
	The values being set below is linked to fields in the HTML, whith the HTML can call
	"""
	feedback = request.form['feedback_button']
	review = request.form['review']
	prediction = request.form['prediction']
	
	inv_label = {'negative':0, 'positive':1}
	y = inv_label[prediction]
	if feedback == 'Incorrect':
		y = int(not(y))  # inverse the label if incorrect is given
	# print('y is :', type(y))
	# print(y)
	# print('review is :', review)  # check the review
	# print(type(review))
	train(review, y) # update the model with the latest data (SGD)
	sqlite_entry(db, review, y)  # add this entry to the database with the 'correct'
	return render_template('thanks.html')

if __name__ == '__main__':
	# app.run(debug=True)
	app.run()



