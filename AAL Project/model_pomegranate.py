import pandas as pd
import numpy as np
import json
from datetime import datetime
from pomegranate import BayesianNetwork
import utilities as util
import main
from six.moves import cPickle as pickle
import utilities as util


def createModel(train, test):
    print("I am in create model")

    header = ['acceleration_mean', 'acceleration_stdev', 'pitch1', 'pitch2', 'pitch3', 'roll1', 'roll2', 'roll3', 'classes', 'total_accel_sensor_1', 'total_accel_sensor_2','total_accel_sensor_4']

    start_time = datetime.now()
    print("Start time: ",start_time)


    model = BayesianNetwork.from_samples(train, algorithm='greedy', state_names=header)

    print("doing model.bake")
    model.bake()

    time = datetime.now() - start_time
    print("Time: ", time)

    predict = test['classes'].tolist()
    test['classes'] = None

    print("Evaluating predict...")
    test = test.to_numpy()
    pred_values = model.predict(test)

    pred_values = [x.item(2) for x in pred_values ]
    main.calculate_accuracy(predict, pred_values)


# MAIN
if __name__ == "__main__":
    # Load train and test dataset
    train, test = main.train_test()
    print("Train and Test loaded successfully!\n")

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    #Creation of a new model
    print('\nStarting creation of a new model...')
    createModel(train,test)
