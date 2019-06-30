from sklearn.naive_bayes import GaussianNB
import numpy as np
import os
from pprint import pprint #serve per fare la print di array e liste in maniera ordinata
import time, calendar, datetime
from datetime import datetime
import csv
import pickle
import pandas as pd
import pickle
from pgmpy.models import BayesianModel, NaiveBayes
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.readwrite import BIFWriter, BIFReader
import warnings ## Remove all warning

warnings.filterwarnings('ignore')

# Carica il dataset dal file csv. Si potrebbe migliorare specificando anche il tipo di ciascuna variabile
def loadDataset():
    filename = 'dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=';')
    x = list(reader)
    dataset = np.array(x[1:])


    results = dataset[:, 18:]
    classes = []
    a = ['sittingdown', 'standingup', 'walking', 'standing', 'sitting']
    for i in range(len(results)):
        classes.append(a.index(results[i]))
    results =np.array(classes)

    dataset = dataset[:, 6:17].astype(np.int)
    print(results)
    print(dataset)


    return dataset, results


def createNB(dataset, results):
    model = GaussianNB()

    model.fit(dataset,results)

    print(model)

    predicted = model.predict([dataset[123300],dataset[79907]])

    print(predicted)

dataset, results = loadDataset()
createBN(dataset, results)
