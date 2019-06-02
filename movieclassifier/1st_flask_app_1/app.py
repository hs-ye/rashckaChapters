# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:06:05 2018
Raschka - Python Machines Learning
Follow along

Chapter 9 - Embedding Machine Learning to Web Apps
@author: yehan
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('first_app.html')

if __name__ == '__main__':
	app.run()



