import pandas as pd
import csv
import numpy as np
from pprint import pprint
import warnings ## Remove all warning
warnings.filterwarnings('ignore')
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
    subjects = ['katia','debora','wallace','jose_carlos']
    for clas in classes:
        dataset_class = dataset_feature[dataset_feature.classes == clas]
        for name in subjects:
            dataset_class_name = dataset_class[dataset_feature.user == name]
            dataset_class_name.drop(columns=['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4'], axis=1, inplace=True)
            dataset_class_name.to_csv('csv/' + clas + '_' + name + '_dataset.csv', sep = ';', index=False)
            sample_split(dataset_class_name,clas,name)


def sample_split(sample,clas,name):
    sample.index = [i for i in range(len(sample)) ]
    l = len(sample)/8

    for i in range(l):
        k = i*8
        if k+8 > len(sample):
            window = sample.iloc[k:len(sample),:]
        else:
            window = sample.iloc[k:k+8,:]

        # appending to the empty data frame the new row
        new_sample = variance_evaluation(window)
        new_sample = split_classes(new_sample)

        user = new_sample['user']
        gender = new_sample['gender']
        age = new_sample['age']
        height = new_sample['height']
        weight = new_sample['weight']
        bmi = new_sample['bmi']
        roll1 = new_sample['roll1']
        pitch1 = new_sample['pitch1']
        roll2 = new_sample['roll2']
        pitch2 = new_sample['pitch2']
        roll3 = new_sample['roll3']
        pitch3 = new_sample['pitch3']
        roll4 = new_sample['roll4']
        pitch4 = new_sample['pitch4']
        #classes = new_sample['classes']
        total_accel_sensor_1 = new_sample['total_accel_sensor_1']
        total_accel_sensor_2 = new_sample['total_accel_sensor_2']
        total_accel_sensor_3 = new_sample['total_accel_sensor_3']
        total_accel_sensor_4 = new_sample['total_accel_sensor_4']
        sittingdown = new_sample['sittingdown']
        standingup = new_sample['standingup']
        walking = new_sample['walking']
        standing = new_sample['standing']
        sitting = new_sample['sitting']

        with open('data_sampled.csv','a') as csvFile:
            row = [user, gender, age, height, weight, bmi, roll1, pitch1, roll2, pitch2, roll3, pitch3, roll4, pitch4, total_accel_sensor_1, total_accel_sensor_2, total_accel_sensor_3, total_accel_sensor_4, sittingdown, standingup, walking, standing, sitting]
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()



# evaluate variance of the eight rows in input
def variance_evaluation(subset):
    new_row = {}

    for field in ['user', 'gender', 'age', 'height', 'weight', 'bmi', 'classes']:
        new_row[field] = subset[field].iloc[0]

    for field in ['roll1', 'pitch1', 'roll2', 'pitch2', 'roll3', 'pitch3','roll4', 'pitch4','total_accel_sensor_1',
                 'total_accel_sensor_2','total_accel_sensor_3','total_accel_sensor_4']:
        new_row[field] = np.var(subset[field])

    return new_row

def split_classes(sample):
    classes = ['sittingdown', 'standingup', 'walking', 'standing', 'sitting']
    classes_ = []
    for value in classes:
        if sample['classes'] == value:
            classes_.append(1)
        else:
            classes_.append(0)

    for i in range(len(classes)):
        sample[classes[i]] = classes_[i]

    return sample

if __name__ == '__main__':
    #Loading of the dataset
    dataset = loadDataset()
    #Evaluation of roll pitch and acceleration vector for any point at any time
    dataset = features_extraction(dataset)
    #Save data into a csv
    dataset.to_csv('csv/measure_dataset.csv', sep = ';', index=False)
    #sample dataset by class
    sample_dataset(dataset)
