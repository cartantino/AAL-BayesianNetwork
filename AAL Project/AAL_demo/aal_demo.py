import numpy as np
import math
from flask import Flask, render_template, request
import io
import sys
import csv
import pickle
import pandas as pd
import random
from operator import itemgetter
from pomegranate import BayesianNetwork

app = Flask(__name__)

with open('example', 'rb') as f:
    example = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html", main=True)

# legge i valori inseriti dall'utente, li aggiunge a window, ritorna a index, richiama main e passa window da stampare
row_to_predict = []
row = []
true_prediction = ""

@app.route('/loadExample', methods=['GET', 'POST'])
def loadExample():
    global row_to_predict
    global true_prediction
    global row

    classes = ['sittingdown','standingup','walking','standing','sitting', 'none']

    true_prediction = request.form.get('class')
    class_to_random = classes.index( true_prediction)

    data = example[0]
    data = data.reindex(sorted(data.columns), axis=1)
    data = data.round(3)

    if class_to_random == 5:
        row_to_predict = data.sample(n=1)
        i = row_to_predict['classes'].values
        true_prediction = classes[i[0]]
    else:
        row_to_predict = data[data['classes']==class_to_random].sample(n=1)

    row = row_to_predict.iloc[0].values
    row = row[3:]

    discrete = example[1]
    row_to_predict = discrete.iloc[row_to_predict.index[0]]

    return render_template(
        'index.html', has_sample=True, main=True,
        sample = row,
        has_predict_button=True,
        true_prediction = true_prediction
        #n_samples = n_samples
    )

@app.route('/prediction')
def predict():
    global row_to_predict

    f = open('model_pomm.txt', "r")
    contents = f.read()
    model = BayesianNetwork.from_json(contents)

    row_to_predict['classes'] = None

    prediction_prob = model.predict_proba(row_to_predict.to_numpy())
    prediction_prob = prediction_prob[2].parameters[0]

    classes = ['sittingdown','standingup','walking','standing','sitting']

    result = []
    for item in prediction_prob.items():
        y = [classes[item[0]], round(item[1],2) ]
        result.append(y)

    result = sorted(result, key=itemgetter(1), reverse=True)

    return render_template(
        'index.html', has_sample=True, main=True,
        has_prediction=True,
        has_predict_button=False,
        sample = row,
        result = result,
        true_prediction = true_prediction
    )

if __name__ == "__main__":
    app.run()
