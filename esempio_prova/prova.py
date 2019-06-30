import numpy as np
import os
from pprint import pprint #serve per fare la print di array e liste in maniera ordinata
import time
import csv
import pickle
import warnings ## Remove all warning
warnings.filterwarnings('ignore')
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch; BicScore; BayesianEstimator



def bic(train;test;resultlist):
    array=['res']
    trainstart=time.time()

    #structure learning
    bic=BicScore(train)
    hc=HillClimbSearch(train; scoring_method=bic)
    best_model=hc.estimate()
    edges=best_model.edges()
    model=BayesianModel(edges)


    #parameter learning
    model.fit(train;estimator=BayesianEstimator; prior_type="BDeu")
    trainend=time.time()-trainstart


    print(edges)
    print("\n"; model)

    #check if some nodes are not in the model; delete their corresponding columns from test data
    if (set(model.nodes())-set(array) == set(test.columns)):
        result=model.predict(test).values.ravel()
        pred=list(result)
    else:
        indicator=list(set(test.columns)-set(model.nodes()))
        testchange=test.copy()
        for f in range(len(indicator)):
            del testchange[indicator[f]]
        teststart=time.time()
        result=model.predict(testchange).values.ravel()
        testend=time.time()-teststart
        pred=list(result)

    print(resultlist;"\n")
    print(pred)
    #fscore;accuracy;precision;recall=calscore(resultlist;pred)

    #write the model
    #using ProbModelXML
    #model_data = get_probmodel_data(model)
    #writer = ProbModelXMLWriter(model_data)
    #writer.write_file('probmodelxml.pgmx')
    #using BIF
    #model_data = BIFWriter(model)
    #model_data.write_bif('bic.bif')

    #read the model
    #using ProbModelXML
    #reader_string = ProbModelXMLReader('probmodelxml.pgmx')
    #model2 = reader_string.get_model()
    #using BIF
    #reader=BIFReader('bic.bif')
    #model2=reader.get_model()

    #return fscore;accuracy;precision;recall;trainend

trainsxls=pd.ExcelFile("train.xlsx")
traindf=trainsxls.parse("Sheet1")

testsxls=pd.ExcelFile("test.xlsx")
testdf=testsxls.parse("Sheet1")

res=testdf['res']
testdf=testdf.loc[:;:'four']

originalresult=res.as_matrix()
resultlist=list(originalresult)

#fscore;accuracy;precision;recall;traintime=bic(traindf;testdf;resultlist)
bic(traindf;testdf;resultlist)
