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
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, K2Score
from pgmpy.readwrite import BIFWriter
#import warnings ## Remove all warning
#warnings.filterwarnings('ignore')
import preprocessing
import hillclimb


'''
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
    a = ['sittingdown', 'standingup', 'walking', 'standing', 'sitting']
    for i in range(len(dataset)):
        classes.append(a.index(dataset[i][18]))
    print(classes)
    input("dd")

    data['classes'] = classes

    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    results = test.loc[:, 'classes'].as_matrix()
    test = test.drop(columns='classes')

    pprint(train)
    pprint(test)
    print(results)

    return train, test, results
'''
'''
def createBN(train,test,resultlist):
    trainstart = datetime.now()
    print("\n\nStart-time: ", trainstart)

    #structure learning
    hc = HillClimbSearch(train, scoring_method=BicScore(train))
    best_model = hc.estimate()
    edges = best_model.edges()

    print(sorted(best_model.nodes()))
    print(sorted(best_model.edges()))

    model = BayesianModel(edges)


    #parameter learning
    model.fit(train, estimator = BayesianEstimator, prior_type = "BDeu")
    print("\nmodel", model)

    trainend = datetime.now()
    print("End-time: ", trainend,"\n\n")

    pred = model.predict(test)
    pred_probs = model.predict_probability(test)

    print("\nresultlist\n",resultlist)
    print("\pred\n",pred)
    print("\pred\n",pred_probs)

    exact = 0
    for i in range(len(resultlist)):
        print("pos:", i, "- expected: ",resultlist[i], "- predicted: ", pred[i][0] )
        if resultlist[i] == pred[i][0]:
            exact += 1

    brumss = float(exact)/float(len(resultlist))
    print("accuracy: ", brumss)

    #using BIF
   
'''

def resultsBN(test,resultlist):
    reader = BIFReader('model_1000.bif')
    model = reader.get_model()

    pred = model.predict(test).as_matrix()
    pred_probs = model.predict_probability(test)

    exact = 0
    for i in range(len(resultlist)):
        print("pos:", i, "- expected: ",resultlist[i], "- predicted: ", pred[i][0] )
        if resultlist[i] == pred[i][0]:
            exact += 1

    brumss = float(exact)/float(len(resultlist))
    print("accuracy: ", brumss)


def print_model(edges):
    print edges
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='b', arrows=True)
    plt.show()

def create_BN_model(data):
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

    print_model(best_model.edges())

    start_time = time.time()
    AAL_model_estimated = BayesianModel(best_model.edges())
    AAL_model_estimated.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
    end_time = time.time()
    pl_time = end_time - start_time

    AAL_model_data = BIFWriter(AAL_model_estimated)
    AAL_model_data.write_bif('Modelli/model_normalized.bif')



    return (AAL_model_estimated , sl_time + pl_time)



if __name__ == "__main__":
    # Load of the dataset preprocessed before
    sampled_dataset = preprocessing.load_data_discrete()

    start_time = datetime.now()
    print("Starting time : "+ str(start_time.hour) + "." + str(start_time.minute) + "." + str(start_time.second))

    #Evaluation of the best model with hill_climb_search, all the data are processed
    best_model, total_time = create_BN_model(sampled_dataset)

    end_time = datetime.now() - start_time
    print("Total time elapsed HC : " + str(end_time.hour) + "." + str(end_time.minute) + "." + str(end_time.second))



    #train, test, resultlist = getAccelometersData()
    #createBN(train,test,resultlist)

    #using BIF
    #reader = BIFReader('model.bif')
    #model = reader.get_model()
