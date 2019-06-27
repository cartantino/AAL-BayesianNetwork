import numpy as np
import os
from pprint import pprint #serve per fare la print di array e liste in maniera ordinata
import time
import csv
import pickle


# Carica il dataset dal file csv. Si potrebbe migliorare specificando anche il tipo di ciascuna variabile
def loadDataset():
    filename = 'dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=';')
    x = list(reader)
    dataset = np.array(x[1:])

    return dataset

# Divide il dataset in training(80%) e test(20%)
def splitDataset(dataset): #
    len = dataset.shape[0]
    training_indexes = np.random.randint(len, size=round(len * 0.8))
    test_indexes = np.random.randint(len, size=round(len * 0.2))
    training, test = dataset[training_indexes,:], dataset[test_indexes,:]

    print(dataset.shape)
    print(training.shape[0] + test.shape[0])



if __name__ == "__main__":
    if os.path.isfile('dataset'):
        print("Pickle 'dataset' trovato.")
        with open ('dataset', 'rb') as fp:
            dataset = pickle.load(fp)
        print("Pickle 'dataset' caricato con successo!")
    else:
        dataset = loadDataset()
        with open('dataset', 'wb') as fp:
            pickle.dump(dataset, fp)
        print("Pickle 'dataset' creato.")
