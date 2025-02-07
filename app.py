from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('SVR_model.pkl', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('home.html', **locals())

@app.route('/predict_api', methods = ['POST', 'GET'])
def predict_pi():
    x1 = float(request.form['x1'])
    x2 = float(request.form['x2'])
    x3 = float(request.form['x3'])
    x4 = float(request.form['x4'])

    result = model.predict([[x1, x2, x3, x4]])
    return render_template('home.html', **locals())
if __name__ == 'main':
    app.run(debug = True)