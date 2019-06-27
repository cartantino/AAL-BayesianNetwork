import numpy as np
import os
from pprint import pprint #serve per fare la print di array e liste in maniera ordinata
import time
import csv
import pickle
import warnings ## Remove all warning
import pandas as pd
from pgmpy.models import BayesianModel

warnings.filterwarnings('ignore')

# Carica il dataset dal file csv. Si potrebbe migliorare specificando anche il tipo di ciascuna variabile
# user;gender;age;how_tall_in_meters;weight;body_mass_index;x1;y1;z1;x2;y2;z2;x3;y3;z3;x4;y4;z4;class
def loadDataset():
    filename = 'dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=';')
    x = list(reader)
    dataset = np.array(x[1:])

    return dataset

# Estrae dal dataset i valori delle coordinate dei vari accelometers
def getAccelometersData(dataset):
    data = pd.DataFrame(data={'x1': [ dataset[i][6] for i in range(len(dataset))  ],
                              'y1': [ dataset[i][7] for i in range(len(dataset))  ],
                              'z1': [ dataset[i][8] for i in range(len(dataset))  ],
                              'x2': [ dataset[i][9] for i in range(len(dataset))  ],
                              'y2': [ dataset[i][10] for i in range(len(dataset)) ],
                              'z2': [ dataset[i][11] for i in range(len(dataset)) ],
                              'x3': [ dataset[i][12] for i in range(len(dataset))  ],
                              'y3': [ dataset[i][13] for i in range(len(dataset))  ],
                              'z3': [ dataset[i][14] for i in range(len(dataset))  ],
                              'x4': [ dataset[i][15] for i in range(len(dataset))  ],
                              'y4': [ dataset[i][16] for i in range(len(dataset)) ],
                              'z4': [ dataset[i][17] for i in range(len(dataset)) ],
                              'class': [ dataset[i][18] for i in range(len(dataset)) ],
                              })
    return(data)

# Divide il dataset in training(80%) e test(20%)
def splitDataset(dataset): #
    len = dataset.shape[0]
    training_indexes = np.random.randint(len, size=round(len * 0.8))
    test_indexes = np.random.randint(len, size=round(len * 0.2))
    training, test = dataset[training_indexes,:], dataset[test_indexes,:]

    print(dataset.shape)
    print(training.shape[0] + test.shape[0])

def createBN(data):
    model = BayesianModel([('x1', 'class'), ('y1', 'class'),('z1', 'class'),
                           ('x2', 'class'), ('y2', 'class'),('z2', 'class'),
                           ('x3', 'class'), ('y3', 'class'),('z3', 'class'),
                           ('x4', 'class'), ('y4', 'class'),('z4', 'class'),])

    pe = ParameterEstimator(model, data)
    print("\n", pe.state_counts('x1'))
    print("\n", pe.state_counts('class'))


if __name__ == "__main__":
    dataset = loadDataset()
