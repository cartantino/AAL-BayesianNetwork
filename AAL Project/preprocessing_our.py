import pandas as pd
import csv
import numpy as np
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
import utilities as util
from sklearn.preprocessing import normalize
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import preprocessing


'''
*************************************************
* ROLL, PITCH AND ACCELERATION VECTOR EVALUATION*
*************************************************
'''
def features_extraction(dataset):
    for i in range(1,5):
        dataset['roll'+str(i)] = pd.Series(180/np.pi*(np.arctan2(dataset['y'+str(i)], dataset['z'+str(i)])))
        dataset['pitch'+str(i)] = pd.Series(180/np.pi*(np.arctan2(-dataset['x'+str(i)], np.sqrt(np.power(dataset['y'+str(i)], 2) + np.power(dataset['z'+str(i)], 2)))))
        dataset['total_accel_sensor_'+str(i)] = pd.Series(np.sqrt(np.power(dataset['x'+str(i)], 2) + np.power(dataset['y'+str(i)], 2) + np.power(dataset['z'+str(i)], 2)))

    return dataset

'''
**************************************************
*SAMPLE DATASET --> 8 ROWS ARE ONE SECOND LECTURE*
**************************************************
'''
def sample_dataset(dataset_with_rpa):
    dataset_sampled = pd.DataFrame(columns=['user','gender','age','height','weight','bmi','roll1','pitch1','roll2','pitch2','roll3','pitch3','roll4','pitch4', 'total_accel_sensor_1', 'total_accel_sensor_2','total_accel_sensor_3','total_accel_sensor_4','classes','acceleration_mean','acceleration_stdev'])
    classes = ['sittingdown','standingup','walking','standing','sitting']
    subjects = ['katia','debora','wallace','jose_carlos']

    dataset_with_rpa.drop(columns=['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4'], axis=1)

    for clas in classes:
        dataset_class = dataset_with_rpa[dataset_with_rpa.classes == clas]
        for name in subjects:
            print(clas, name)
            dataset_class_name = dataset_class[dataset_with_rpa.user == name]

            #sample split
            dataset_class_name.index = [i for i in range(len(dataset_class_name))]
            l = int(len(dataset_class_name)/8)

            for i in range(l):
                k = i*8
                if k+8 > len(dataset_class_name):
                    window = dataset_class_name.iloc[k:len(dataset_class_name),:]
                else:
                    window = dataset_class_name.iloc[k:k+8,:]

                new_sample = variance_evaluation(window)
                new_sample = split_classes(new_sample)
                dataset_sampled = dataset_sampled.append(new_sample, ignore_index=True)

    dataset_sampled['classes'] = dataset_sampled['classes'].astype(int)

    return dataset_sampled

'''
***********************************************
VARIANCE-MEAN-STD EVALUATION OF THE WINDOW    *
***********************************************
'''
def variance_evaluation(window):
    new_row = {}

    for field in ['user', 'gender', 'age', 'height', 'weight', 'bmi', 'classes']:
        new_row[field] = window[field].iloc[0]

    for field in ['roll1', 'pitch1', 'roll2', 'pitch2', 'roll3', 'pitch3','roll4', 'pitch4','total_accel_sensor_1','total_accel_sensor_2','total_accel_sensor_3','total_accel_sensor_4']:
        new_row[field] = np.mean(window[field])

    new_row['acceleration_mean'] = np.mean([new_row['total_accel_sensor_1'],new_row['total_accel_sensor_2'],new_row['total_accel_sensor_3'],new_row['total_accel_sensor_4']])
    new_row['acceleration_stdev'] = np.std([new_row['total_accel_sensor_1'],new_row['total_accel_sensor_2'],new_row['total_accel_sensor_3'],new_row['total_accel_sensor_4']])

    return new_row

'''
******************************************
SPLIT DATASET INTO CLASSES --> USERS     *
******************************************
'''
def split_classes(sample):

    sample['classes'] = sample['classes'].replace('sittingdown','0')
    sample['classes'] = sample['classes'].replace('standingup','1')
    sample['classes'] = sample['classes'].replace('walking','2')
    sample['classes'] = sample['classes'].replace('standing','3')
    sample['classes'] = sample['classes'].replace('sitting','4')

    return sample

'''
************************************************
*   FEATURE SELECTION AND DISCRETIZATION       *
************************************************
'''
def feature_selection_discretization(dataset_sampled):
    classes = ['sittingdown','standingup','walking','standing','sitting']
    subjects = ['katia','debora','wallace','jose_carlos']
    dataset_discretized = pd.DataFrame(columns=['roll1','pitch1','roll2','pitch2','roll3','pitch3','total_accel_sensor_1','total_accel_sensor_3','roll4','pitch4','total_accel_sensor_2','total_accel_sensor_4','classes','acceleration_mean','acceleration_stdev'])

    for clas in range(5):
        data_class = dataset_sampled[dataset_sampled.classes == clas]

        for name in subjects:
            data_class_name = data_class[dataset_sampled.user == name]
            data_class_name.drop(columns=['user','gender','age','height','weight','bmi'], axis=1, inplace=True)

            data_class_name = discretize(data_class_name)
            dataset_discretized = dataset_discretized.append(data_class_name)

    return dataset_discretized

'''
********************************************************************
* COLUMNS DISCRETIZATION --- TODO : EVALUATE HOW MUCH BINS AND WHY *
********************************************************************
'''
def discretize(data):
    m = 10

    classi = data['classes']
    data = data.drop('classes', axis=1)

    columns=['roll1','pitch1','roll2','pitch2','roll3','pitch3','total_accel_sensor_1','total_accel_sensor_2','total_accel_sensor_4','acceleration_mean','acceleration_stdev','total_accel_sensor_3','roll4','pitch4']

    # Normalizzazione
    data[columns] = data[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Discretizzazione
    for column in columns:
        data[column] = pd.cut(data[column], m , labels=False)

    data['classes'] = classi

    return data

def split_classes_columns(sample):

    sample['classes'] = sample['classes'].replace(0,'sittingdown')
    sample['classes'] = sample['classes'].replace(1,'standingup')
    sample['classes'] = sample['classes'].replace(2,'walking')
    sample['classes'] = sample['classes'].replace(3,'standing')
    sample['classes'] = sample['classes'].replace(4,'sitting')

    classes = ['sittingdown', 'standingup', 'walking', 'standing', 'sitting']

    for clas in classes:
        classes_ = []

        for index, row in sample.iterrows():
            if row['classes'] == clas:
                classes_.append(1)
            else:
                classes_.append(0)

        for i in range(len(classes)):
            sample[clas] = classes_
            sample[clas] = sample[clas].astype(int)

    sample.drop(columns=['classes'], axis=1, inplace=True)

    return sample


if __name__ == "__main__":
    # Load original dataset
    print("Uploading dataset..")
    dataset = util.load_dataset()


    # Cleaning of dataset
    dataset = dataset.drop(dataset[dataset.x1 > 100].index)
    dataset = dataset.drop(dataset[dataset.x1 < -100].index)
    dataset = dataset.drop(dataset[dataset.y1 > 200].index)
    dataset = dataset.drop(dataset[dataset.z1 < -250].index)
    dataset = dataset.drop(dataset[dataset.x2 > 200].index)
    dataset = dataset.drop(dataset[dataset.x4 < -400].index)
    dataset = dataset.drop(dataset[dataset.y4 > 0].index)
    dataset = dataset.drop(dataset[dataset.y4 < -200].index)
    dataset = dataset.drop(dataset[dataset.z4 < -220].index)

    # Extract features from the dataset
    print("Extracting useful features from the dataset..")
    dataset = features_extraction(dataset)
    print('Dataset with extracted features')
    print(dataset[:100])

    # Sample dataset
    print("Sampling dataset...")
    dataset_sampled = sample_dataset(dataset)
    print(dataset_sampled)

    # Save dataframe as pickle
    print('Saving dataset_sampled to pickle object...')
    util.to_pickle(dataset_sampled, 'dataset_sampled')

    with open('sampled_dataset.csv', 'w', newline='') as c:
       dataset_sampled.to_csv(c, header=True, index=False)
    c.close()
    print('\n Dataset sampled')

    # Feature selection (Mark Hall's algorithm, reference paper)
    print('Feature selection...')
    dataset_discrete = feature_selection_discretization(dataset_sampled)

    with open('discrete_dataset.csv', 'w', newline='') as c:
        dataset_discrete.to_csv(c, header=True, index=False)
    c.close()

    print("Discrete dataset\n")
    print(dataset_discrete)

    # Save dataframe as pickle
    print('Saving dataset_discrete to pickle object...')
    util.to_pickle(dataset_discrete, 'pickle_data_discrete')

    print('Preprocessing has finished successfully.')
