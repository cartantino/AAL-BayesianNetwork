#MAIN PROVVISORIO
import utilities as util
import numpy as np
import os
import csv
import pickle
import pandas as pd
import pickle
import time, calendar, datetime
from pprint import pprint
from datetime import datetime
from pgmpy.inference.ExactInference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, K2Score
from pgmpy.readwrite import BIFWriter, BIFReader
import warnings ## Remove all warning
warnings.filterwarnings('ignore')
import hillclimb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels



'''
***************************************************
*    SPLIT DATA DISCRETE INTO TRAIN AND TEST      *
*    TRAIN --> 80%                                *
*    TEST --> 20%                                 *
*    ALL THE CLASSES ARE BALANCED IN EACH FOLDER  *
***************************************************
'''
def train_test():
    if os.path.isfile('pickle_train') and os.path.isfile('pickle_test'):
        test = util.load_pickle('pickle_test')
        train  = util.load_pickle('pickle_train')

        test.index = [i for i in range(len(test))]
        train.index = [i for i in range(len(train))]

        return train, test
    else:
        #Load of the dataset preprocessed before
        dataset = util.load_pickle('pickle_data_discrete')

        header = ['acceleration_mean', 'acceleration_stdev', 'pitch1', 'pitch2', 'pitch3', 'roll1', 'roll2', 'roll3',
                    'classes', 'total_accel_sensor_1', 'total_accel_sensor_2',
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

        for i in range(5):
            #find in dataset one class at once
            c = dataset.loc[dataset['classes'] == i]

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

        #write train and test in a pickle
        util.to_pickle(total_train, 'pickle_train')
        util.to_pickle(total_test, 'pickle_test')

        total_train.index = [i for i in range(len(total_train))]
        total_test.index = [i for i in range(len(total_test))]

        return total_train, total_test

'''
***********************************************************************
*   CREATE A BAYESIAN NETWORK WITH HILL CLIMB OR FROM SPECIFIED EDGES *
***********************************************************************
'''
def create_BN_model(data):
    #structure learning
    print("Structure learning")
    start_time = datetime.now()
    print("Start time: ",start_time)

    #DECOMMENT TO CREATE A MODEL WITH THE HILL CLIMB ALGORITHM
    hc = HillClimbSearch(data)

    best_model = hc.estimate()
    print(best_model.edges())
    edges = best_model.edges()

    model = BayesianModel(edges)

    print('Fitting the model...')

    # Evaluation of cpds using Maximum Likelihood Estimation
    model.fit(data)

    end_time = datetime.now()
    print("End time: ",end_time)

    model_write = BIFWriter(model)
    model_write.write_bif('model_pgmpy.bif')

    if model.check_model():
        print("Your network structure and CPD's are correctly defined. The probabilities in the columns sum to 1. Hill Climb worked fine!")
    else:
        print("not good")
    return (model , end_time-start_time)


def inference(test, model):
    predict = test['classes'].tolist()
    test.drop(['classes'], axis=1, inplace=True)
    pred_values = model.predict(test)
    pred_values = pred_values['classes'].tolist()

    calculate_accuracy(predict, pred_values)

def calculate_accuracy(test, pred):
    class_names = ["sittingdown", "standingup", "walking", "standing", "sitting"]

    # Calculate accuracy percentage
    accuracy = accuracy_score(test, pred)
    print("Accuracy: ", accuracy,"%")

    # Calculate f1-score
    f1 = f1_score(test, pred, average='macro')
    print("F1-score: ", f1)

     # Calculate precision
    precision = precision_score(test,pred, average = 'macro')
    print("Precision: ", precision)

     # Calculate recall
    recall = recall_score(test, pred, average = 'macro')
    print("Recall: ", recall)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(test, pred, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(test, pred, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('\nConfusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



if __name__ == "__main__":
    # Load train and test dataset
    train, test = train_test()
    print("Train and Test loaded successfully!\n")

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    # Creation of a new model
    print('\nStarting creation of a new model...')
    model, total_time = create_BN_model(train)
    print(total_time)

    # Load of the model we want to use to make inference
    print('Loading the model...')
    reader=BIFReader('model_pgmpy.bif')
    model=reader.get_model()
    if model.check_model():
        print("Your network structure and CPD's are correctly defined. The probabilities in the columns sum to 1. Hill Climb worked fine!")
    else:
        print("not good")

    # Inference
    print('\nStarting inference...')
    inference(test, model)

    print('Inference done!')
