import numpy as np
import os
import csv
import pickle
import pandas as pd
import pickle
import time, calendar, datetime
from pgmpy.inference.ExactInference import VariableElimination
from pprint import pprint #serve per fare la print di array e liste in maniera ordinata
from datetime import datetime
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, K2Score
from pgmpy.readwrite import BIFWriter, BIFReader
#import warnings ## Remove all warning
#warnings.filterwarnings('ignore')
import preprocessing
import hillclimb
from sklearn.model_selection import train_test_split


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
    
    # Check if the model is ok, see documentation for further information
    if best_model.check_model():
        print("Your network structure and CPD's are correctly defined. The probabilities in the columns sum to 1. Hill Climb worked fine!")
    edges = best_model.edges()
    print_model(best_model.edges())
    AAL_model_data = BIFWriter(AAL_model_estimated)
    AAL_model_data.write_bif('Modelli/model_afterclean.bif')

    return (AAL_model_estimated , sl_time + pl_time)

#function to train and test discrete csv
def train_test(dataset):
    classes=['sitting', 'sittingdown', 'standing', 'standingup', 'walking']

    header = ['acceleration_mean', 'acceleration_stdev', 'pitch1', 'pitch2', 'pitch3', 'roll1', 'roll2', 'roll3',
                'sitting', 'sittingdown', 'standing', 'standingup', 'walking', 'total_accel_sensor_1', 'total_accel_sensor_2',
                'total_accel_sensor_4']

    #write header in train and test csv
    with open('train_dataset.csv', "w", newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(header)
    csvFile.close()

    with open('test_dataset.csv', "w", newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(header)
    csvFile.close()

    total_test = dataset[0:0]
    total_train = dataset[0:0]

    for cl in classes:
        #find in dataset one class at once
        c = dataset.loc[dataset[cl] == 1]

        #training and testing of one class: 80% training, 20% testing of each class
        train, test = train_test_split(c, test_size=0.2)

        total_test = total_test.append(test)
        total_train = total_train.append(train)

        #append results in two csv, train and test
        with open('train_dataset.csv', 'a', newline='') as csvFile:
            train.to_csv(csvFile, header=False, index=False)
        csvFile.close()

        with open('test_dataset.csv', 'a', newline='') as csvFile:
            test.to_csv(csvFile, header=False, index=False)
        csvFile.close()

    print(len(total_train), len(total_test))

    return total_train, total_test



def cpd_estimation(model, train):
    print("Estimation of the cpds of the model")
    print(model.edges())
    model = BayesianModel(model.edges())
    model.fit(train, estimator=BayesianEstimator, prior_type="BDeu")
    for node in model.nodes():
       print(model.get_cpds(node))
    cpds = model.get_cpds()
    #PRINT OF THE CPDs of the model
    for cpd in model.get_cpds():
        pprint("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)

    model_write = BIFWriter(model)
    model_write.write_bif('Modelli/model_cpd.bif')

    return cpds, model

def inference(train, test, model):
    
    # Evaluation of the cpd of the model
    cpds, model_inf = cpd_estimation(model, train)
    #Associate cpds to the model
    #model_inf.add_cpds(cpds)
   
    # Creating the inference object of the model
    #AAL_inference = VariableElimination(model_inf)  

    # # If we have some evidence for the network we can simply pass it
    # # as an argument to the query method in the form of 
    #print(AAL_inference.query(variables=['sittingdown']))#,evidence={''variable':value} 

    #predict function uses exact inference on nodes wich are not given in the query
    prediction = model.predict(test)#.values.ravel()
    print(prediction)



if __name__ == "__main__":
    # Load of the dataset preprocessed before
    #discrete_dataset = preprocessing.load_data_discrete()

    #Decommment these lines to create a new model
    #search for the best model using Hill Climb Algorithm, for further information look at the documentation
    '''
    start_time = datetime.now()
    print("Starting time : "+ str(start_time.hour) + "." + str(start_time.minute) + "." + str(start_time.second))
    model, total_time = create_BN_model(discrete_dataset)
    end_time = datetime.now() - start_time
    print(str(end_time))
    '''

    #Splitting dataset into train and test 80% - 20%
    #train, test = train_test(discrete_dataset)

    #create a csv for test_dataset without 5 classes
    '''removed_column = test.drop(['sitting', 'sittingdown', 'standing', 'standingup', 'walking'], axis=1)
   
    with open('test_inference.csv', "a", newline='') as csvFile:
        removed_column.to_csv(csvFile, index=False)
    csvFile.close()'''

    #Load of the model we want to use to make inference
    reader=BIFReader('Modelli/model_afterclean.bif')
    model=reader.get_model()
    if model.check_model():
        print "Your network structure and CPD's are correctly defined. The probabilities in the columns sum to 1. Hill Climb worked fine!"
    else:
        print "not good" 

    test = preprocessing.load_test_inference()
    train  = preprocessing.load_training()

    #inference
    inference(train,test,model)