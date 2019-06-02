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

app = Flask(__name__)

class HelloForm(Form):
	sayhello = TextAreaField('', [validators.DataRequired()])


@app.route('/')
def index():
	form = HelloForm(request.form)
	return render_template('first_app.html', form=form)

# the post method transports data from form to server
@app.route('/hello', methods=['POST'])  
def hello():
	form = HelloForm(request.form)
	if request.method == 'POST' and form.validate():
		name = request.form['sayhello']
		return render_template('hello.html', name=name)
	return render_template('first_app.html', form=form)


if __name__ == '__main__':
	app.run(debug=True)



