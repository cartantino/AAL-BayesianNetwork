#Utilities module
import pandas as pd
import csv
import numpy as np
from six.moves import cPickle as pickle

# Store a variable into pickle object
def to_pickle(dataframe, path_pickle):
    with open(path_pickle, 'wb') as f:
        pickle.dump(dataframe, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(str(path_pickle) + " saved!")

# Load a pickle object into a variable
def load_pickle(path_pickle):
    with open(path_pickle, 'rb') as f:
        model = pickle.load(f)
        print(str(path_pickle) + " loaded!")
    return model

# Load dataset.csv
def load_dataset():
    filename = 'dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=';')
    x = list(reader)
    dataset = np.array(x[1:])
    dataset = pd.DataFrame(data={
                            'user':[dataset[i][0]       for i in range(len(dataset))],
                            'gender':[dataset[i][1]     for i in range(len(dataset))],
                            'age':[int(dataset[i][2])   for i in range(len(dataset))],
                            'height':[float(dataset[i][3]) for i in range(len(dataset))],
                            'weight':[float(dataset[i][4]) for i in range(len(dataset))],
                            'bmi':[float(dataset[i][5]) for i in range(len(dataset))],
                            'x1': [ int(dataset[i][6]) for i in range(len(dataset)) ],
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
                            'z4': [ int(dataset[i][17]) for i in range(len(dataset)) ],
                            'classes':[dataset[i][18]   for i in range(len(dataset))]})
    return dataset

# Load dataset_sampled.csv
def load_data_sampled():
    filename = 'sampled_dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter = ',')
    x = list(reader)
    dataset = np.array(x[1:])
    dataset = pd.DataFrame(data = {
                        'user' : [dataset[i][0]       for i in range(len(dataset))],
                        'gender' : [dataset[i][1]       for i in range(len(dataset))],
                        'age' : [int(dataset[i][2])   for i in range(len(dataset))],
                        'height' : [float(dataset[i][3]) for i in range(len(dataset))],
                        'weight' : [float(dataset[i][4]) for i in range(len(dataset))],
                        'bmi' : [float(dataset[i][5]) for i in range(len(dataset))],
                        'roll1' : [float(dataset[i][6]) for i in range(len(dataset))],
                        'pitch1' : [float(dataset[i][7]) for i in range(len(dataset))],
                        'roll2' : [float(dataset[i][8]) for i in range(len(dataset))],
                        'pitch2' : [float(dataset[i][9]) for i in range(len(dataset))],
                        'roll3' : [float(dataset[i][10]) for i in range(len(dataset))],
                        'pitch3' : [float(dataset[i][11]) for i in range(len(dataset))],
                        'roll4' : [float(dataset[i][12]) for i in range(len(dataset))],
                        'pitch4' : [float(dataset[i][13]) for i in range(len(dataset))],
                        'total_accel_sensor_1' : [float(dataset[i][14]) for i in range(len(dataset))],
                        'total_accel_sensor_2' : [float(dataset[i][15]) for i in range(len(dataset))],
                        'total_accel_sensor_3' : [float(dataset[i][16]) for i in range(len(dataset))],
                        'total_accel_sensor_4' : [float(dataset[i][17]) for i in range(len(dataset))],
                        'classes' : [dataset[i][18] for i in range(len(dataset))],
                        'acceleration_mean' : [float(dataset[i][19]) for i in range(len(dataset))],
                        'acceleration_stdev' : [float(dataset[i][20]) for i in range(len(dataset))]})

    return dataset

# Load data_discrete.csv
def load_data_discrete():
    filename = 'data_discrete.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter = ',')
    x = list(reader)
    dataset = np.array(x[1:])
    dataset = pd.DataFrame(data = {
                        'acceleration_mean' : [int(dataset[i][0]) for i in range(len(dataset))],
                        'acceleration_stdev' : [int(dataset[i][1]) for i in range(len(dataset))],
                        'roll1' : [int(dataset[i][2]) for i in range(len(dataset))],
                        'pitch1' : [int(dataset[i][3]) for i in range(len(dataset))],
                        'roll2' : [int(dataset[i][4]) for i in range(len(dataset))],
                        'pitch2' : [int(dataset[i][5]) for i in range(len(dataset))],
                        'roll3' : [int(dataset[i][6]) for i in range(len(dataset))],
                        'pitch3' : [int(dataset[i][7]) for i in range(len(dataset))],
                        'classes' : [dataset[i][18] for i in range(len(dataset))],
                        'total_accel_sensor_1' : [int(dataset[i][19]) for i in range(len(dataset))],
                        'total_accel_sensor_2' : [int(dataset[i][20]) for i in range(len(dataset))],
                        'total_accel_sensor_4' : [int(dataset[i][21]) for i in range(len(dataset))]})

    return dataset

# Load data_discrete.csv
def load_test():
    filename = 'test_dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter = ';')
    x = list(reader)
    dataset = np.array(x[1:])
    dataset = pd.DataFrame(data={
        'acceleration_mean' : [int(dataset[i][0]) for i in range(len(dataset))],
        'acceleration_stdev' : [int(dataset[i][1]) for i in range(len(dataset))],
        'roll1' : [int(dataset[i][2]) for i in range(len(dataset))],
        'pitch1' : [int(dataset[i][3]) for i in range(len(dataset))],
        'roll2' : [int(dataset[i][4]) for i in range(len(dataset))],
        'pitch2' : [int(dataset[i][5]) for i in range(len(dataset))],
        'roll3' : [int(dataset[i][6]) for i in range(len(dataset))],
        'pitch3' : [int(dataset[i][7]) for i in range(len(dataset))],
        'classes' : [dataset[i][18] for i in range(len(dataset))],
        'total_accel_sensor_1' : [int(dataset[i][19]) for i in range(len(dataset))],
        'total_accel_sensor_2' : [int(dataset[i][20]) for i in range(len(dataset))],
        'total_accel_sensor_4' : [int(dataset[i][21]) for i in range(len(dataset))]})

    return dataset

# Load train_dataset.csv
def load_train():
    filename = 'train_dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter = ',')
    x = list(reader)
    dataset = np.array(x[1:])
    dataset = pd.DataFrame(data={
                'acceleration_mean' : [int(dataset[i][0]) for i in range(len(dataset))],
                'acceleration_stdev' : [int(dataset[i][1]) for i in range(len(dataset))],
                'roll1' : [int(dataset[i][2]) for i in range(len(dataset))],
                'pitch1' : [int(dataset[i][3]) for i in range(len(dataset))],
                'roll2' : [int(dataset[i][4]) for i in range(len(dataset))],
                'pitch2' : [int(dataset[i][5]) for i in range(len(dataset))],
                'roll3' : [int(dataset[i][6]) for i in range(len(dataset))],
                'pitch3' : [int(dataset[i][7]) for i in range(len(dataset))],
                'classes' : [dataset[i][18] for i in range(len(dataset))],
                'total_accel_sensor_1' : [int(dataset[i][19]) for i in range(len(dataset))],
                'total_accel_sensor_2' : [int(dataset[i][20]) for i in range(len(dataset))],
                'total_accel_sensor_4' : [int(dataset[i][21]) for i in range(len(dataset))]})

    return dataset
