import numpy as np
import os
from pprint import pprint #serve per fare la print di array e liste in maniera ordinata
import time, calendar, datetime
from datetime import datetime
import csv
import pickle
import warnings ## Remove all warning
import pandas as pd
import pickle
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
warnings.filterwarnings('ignore')

# Carica il dataset dal file csv. Si potrebbe migliorare specificando anche il tipo di ciascuna variabile
def loadDataset():
    filename = 'dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=';')
    x = list(reader)
    dataset = np.array(x[1:])

    return dataset

# Estrae dal dataset i valori delle coordinate dei vari accelerometri e divide il dataset in training(80%) e test(20%)
def getAccelometersData():
    dataset = loadDataset()

    data = pd.DataFrame(data={'x1': [ int(dataset[i][6]) for i in range(len(dataset))  ],
                              'y1': [ int(dataset[i][7]) for i in range(len(dataset))  ],
                              'z1': [ int(dataset[i][8]) for i in range(len(dataset))  ],
                              #'x2': [ int(dataset[i][9]) for i in range(len(dataset))  ],
                              #'y2': [ int(dataset[i][10]) for i in range(len(dataset)) ],
                              #'z2': [ int(dataset[i][11]) for i in range(len(dataset)) ],
                              #'x3': [ int(dataset[i][12]) for i in range(len(dataset))  ],
                              #'y3': [ int(dataset[i][13]) for i in range(len(dataset))  ],
                              #'z3': [ int(dataset[i][14]) for i in range(len(dataset))  ],
                              #'x4': [ int(dataset[i][15]) for i in range(len(dataset))  ],
                              #'y4': [ int(dataset[i][16]) for i in range(len(dataset)) ],
                              #'z4': [ int(dataset[i][17]) for i in range(len(dataset)) ],
                              'classes': [ dataset[i][18] for i in range(len(dataset)) ]})

    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    results = test.loc[:, 'classes'].as_matrix()
    test = test.drop(columns='classes')

    return train, test, results

def createBN(train,test,resultlist):
    trainstart = datetime.now()
    print("Start-time: ", trainstart)

    #structure learning
    bic = BicScore(train)
    hc = HillClimbSearch(train, scoring_method=BicScore(train))
    best_model = hc.estimate()
    edges = best_model.edges()

    print(sorted(best_model.nodes()))
    print(sorted(best_model.edges()))

    model = BayesianModel(edges)

    print("\nedges: ", edges)

    #parameter learning
    model.fit(train, estimator = BayesianEstimator, prior_type = "BDeu")
    print("\nmodel", model)

    trainend = datetime.now()
    print("End-time: ", trainend)

    #using BIF
    #model_data = BIF.BIFWriter(model)
    #model_data.write_bif('model.bif')


if __name__ == "__main__":
    train, test, resultlist = getAccelometersData()
    createBN(train,test,resultlist)

    #using BIF
    #reader = BIFReader('best_model.bif')
    #model = reader.get_model()
