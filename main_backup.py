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
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, MaximumLikelihoodEstimator
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

    return dataset

a = ['sittingdown', 'standingup', 'walking', 'standing', 'sitting']
# Estrae dal dataset i valori delle coordinate dei vari accelerometri e divide il dataset in training(80%) e test(20%)
def getAccelometersData():
    dataset = loadDataset()

    data = pd.DataFrame(data={'x1': [ int(dataset[i][6]) for i in range(len(dataset)) ],
                              'y1': [ int(dataset[i][7]) for i in range(len(dataset)) ],
                              'z1': [ int(dataset[i][8]) for i in range(len(dataset)) ],
                              'x2': [ int(dataset[i][9]) for i in range(len(dataset)) ],
                              'y2': [ int(dataset[i][10]) for i in range(len(dataset)) ],
                              'z2': [ int(dataset[i][11]) for i in range(len(dataset)) ],
                              'x3': [ int(dataset[i][12]) for i in range(len(dataset)) ],
                              'y3': [ int(dataset[i][13]) for i in range(len(dataset)) ],
                              'z3': [ int(dataset[i][14]) for i in range(len(dataset)) ],
                              'x4': [ int(dataset[i][15]) for i in range(len(dataset)) ],
                              'y4': [ int(dataset[i][16]) for i in range(len(dataset)) ],
                              'z4': [ int(dataset[i][17]) for i in range(len(dataset)) ]})

    classes = []

    for i in range(len(dataset)):
        classes.append(a.index(dataset[i][18]))

    data['classes'] = classes

    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]

    train.index = [i for i in range(len(train)) ]
    test.index = [i for i in range(len(test)) ]

    results = test.loc[:, 'classes'].as_matrix()
    test = test.copy()
    test.drop(columns='classes', axis=1, inplace=True)

    return train, test, results

def createBN(data):
    print "\n\nStart-time: ", datetime.now()

    model = BayesianModel([('classes', 'x1'), ('classes', 'y1'), ('classes', 'z1'),
                           ('classes', 'x2'), ('classes', 'y2'), ('classes', 'z2'),
                           ('classes', 'x3'), ('classes', 'y3'), ('classes', 'z3'),
                           ('classes', 'x4'), ('classes', 'y4'), ('classes', 'z4')])

    #parameter learning
    print "edges", sorted(model.edges())
    model.fit(data, estimator = MaximumLikelihoodEstimator)
    print "\nmodel", model

    print "End-time: ", datetime.now(),"\n\n"

    print a
    cpd_classes = MaximumLikelihoodEstimator(model, data).estimate_cpd('classes')
    print cpd_classes

    model_data = BIFWriter(model)
    model_data.write_bif('model_2.bif')

    return model

def predictBN(model, test, resultlist):
    #print test

    print "Prova predict"
    pred = model.predict( test)

    print pred


if __name__ == "__main__":
    data, test, resultlist = getAccelometersData()

    if os.path.isfile('modelww_2.bif'):
        print "model_2.bif trovato!"
        #using BIF
        reader = BIFReader('model_2.bif')
        model = reader.get_model()
    else:
        print "Creazione model_2.bif!"
        mmodel = createBN(data)

    predictBN(model, test, resultlist)
