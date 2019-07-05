import numpy as np
import os
import csv
import pickle
import pandas as pd
import pickle
import time, calendar, datetime
from pprint import pprint #serve per fare la print di array e liste in maniera ordinata
from datetime import datetime
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, K2Score, MaximumLikelihoodEstimator
from pgmpy.readwrite import BIFWriter
#import warnings ## Remove all warning
#warnings.filterwarnings('ignore')
import preprocessing
import hillclimb
from sklearn.model_selection import train_test_split



def dataset_norm_discr(dataset):
    classes_dataset = dataset['classes']
    dataset.drop(columns=['user','gender','age','height','weight','bmi','classes'], axis=1, inplace=True)

    dataset=(dataset-dataset.min())/(dataset.max()-dataset.min())

    n=10
    for column in dataset:
        dataset[column] = pd.Series(pd.cut(x=dataset[column], bins=n, labels=list(range(n))))

    dataset['classes'] = classes_dataset
    classes=['sitting', 'sittingdown', 'standing', 'standingup', 'walking']
    for i in range(0,5):
        dataset.loc[(dataset['classes'] == classes[i])] = i

    return dataset

#function to train and test discrete csv
def train_test(dataset):
    classes=['sitting', 'sittingdown', 'standing', 'standingup', 'walking']

    for cl in range(0,5):
        #find in dataset one class at once
        c = dataset.loc[dataset['classes'] == cl]
        #training and testing of one class: 80% training, 20% testing of each class
        train, test = train_test_split(c, test_size=0.2)

    return train, test

def print_model(edges):
    print(edges)
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows=True)
    plt.show()

def calculate_edges(data):
    #structure learning
    print("Structure learning")
    start_time = time.time()

    #data2 = pd.DataFrame(np.random.randint(0, 10, size=(1000, 13)), columns=list('ABCDEFGHILMNO'))

    hc = hillclimb.HillClimbSearch(data)
    #hc = HillClimbSearch(data, scoring_method= BicScore(data))

    best_model = hc.estimate()
    print(hc.scoring_method)
    print(best_model.edges())
    end_time = time.time()
    sl_time = end_time - start_time
    print("execution time in seconds:{}".format(sl_time))

    # Check if the model is ok, see documentation for further information
    best_model.check_model()
    edges = best_model.edges()
    print_model(best_model.edges())

def BN_model1(train, test):
    print 'Fitting model, starting time: ', str(datetime.now())

    edges = [('x1', 'classes'),('x2', 'classes'),('x3', 'classes'),('x4', 'classes'), ('y1', 'classes'),('y2', 'classes'),('y3', 'classes'),('y4', 'classes'),('z1', 'classes'),('z2', 'classes'),('z3', 'classes'),('z4', 'classes')]

    input("ciao")

    model = BayesianModel(edges)
    model.fit(train,  estimator=BayesianEstimator, prior_type="BDeu")

    print 'Fitting model, ending time: ', str(datetime.now())
    #print 'model cpds:', model.get_cpds()
    print model.nodes()
    for node in ['y2', 'z1', 'y1', 'classes', 'z2', 'x2', 'x3', 'y3', 'x1', 'z4', 'y4', 'x4', 'z3']:
        print node,model.get_cpds(node)

    #print 'test: ', test
    test.drop(columns=['classes'], axis=1, inplace=True)
    #print 'test: ', test


    #predicted = model.predict(test)
    #print predicted






if __name__ == "__main__":
    dataset = preprocessing.load_dataset()
    dataset = dataset_norm_discr(dataset)

    train, test = train_test(dataset)
    #calculate_edges(dataset)
    BN_model1(train, test)
