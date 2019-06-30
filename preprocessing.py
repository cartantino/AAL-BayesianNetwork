import pandas as pd
import csv
import numpy as np
from pprint import pprint
# PREPROCESSING


# Carica il dataset dal file csv. Si potrebbe migliorare specificando anche il tipo di ciascuna variabile
def loadDataset():
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


def features_extraction(dataset):

    for i in range(1,5):
        dataset['roll'+str(i)] = pd.Series(180/np.pi*(np.arctan2(dataset['y'+str(i)], dataset['z'+str(i)])))
        dataset['pitch'+str(i)] = pd.Series(180/np.pi*(np.arctan2(-dataset['x'+str(i)], np.sqrt(np.power(dataset['y'+str(i)], 2) + np.power(dataset['z'+str(i)], 2)))))
        dataset['total_accel_sensor_'+str(i)] = pd.Series(np.sqrt(np.power(dataset['x'+str(i)], 2) + np.power(dataset['y'+str(i)], 2) + np.power(dataset['z'+str(i)], 2)))    
    return dataset


def sample_dataset(dataset_feature):
    classes = ['sittingdown','standingup','walking','standing','sitting']
    for clas in classes:
        dataset_class = dataset_feature[dataset_feature.classes == clas]
        dataset_class.to_csv('csv/' + clas + '_dataset.csv', sep = ';', index=False)
        
    

if __name__ == '__main__':
    #Loading of the dataset
    dataset = loadDataset()
    #Evaluation of roll pitch and acceleration vector for any point at any time
    dataset = features_extraction(dataset)
    #Save data into a csv
    dataset.to_csv('csv/measure_dataset.csv', sep = ';', index=False)
    #sample datasetby class
    sample_dataset(dataset)
    
