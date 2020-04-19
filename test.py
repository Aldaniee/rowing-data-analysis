#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:54:29 2020

@author: tylerbrennan
"""

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split

FILE = 'rowingdata.csv'
LABEL = 'BoatPlacementImprovement'
TEST_PERCENT = .2
TOP_N = 3
UNKNOWN = '?'
NEG = '-'


def att_arr(data):
    attributes = []
    for att in data:
        if att != LABEL:
            attributes.append(att)
    return attributes

def median(data):
    for att in data:
        median_vals = []
        replace = []
        for i in range(len(data[att])):
            val = data[att][i]
            if str(val) != UNKNOWN and str(val).find(NEG) != 0:
                val = float(val)
                data[att][i] = val
                median_vals.append(val)
            elif str(val).find(NEG) == 0:
                val = str(val)
                val = float(val[1:]) * -1   # There was an issue with negative values being read as strings
                data[att][i] = val
                median_vals.append(val)
            else:
                replace.append(i)
        for i in replace:
            data[att][i] = np.median(replace)
        data[att] = data[att].astype('float64') # Avoids issue of all values being converted to ints

def scale(data, attributes):
    for att in data:
        min_data = float(data[att].min())
        max_data = float(data[att].max()) - min_data
        for i in range(len(data[att])):
            val = (data[att][i] - min_data) / max_data
            data[att][i] = val
        if att == LABEL:
            count_arr = np.unique(data[LABEL])
            print(count_arr)
            print(count_arr[0])
            for i in range(len(data[LABEL])):
                data[LABEL][i] = np.where(count_arr == data[LABEL][i])[0][0]

data = pd.read_csv(FILE)
attributes = att_arr(data)
pd.set_option('mode.chained_assignment', None)
median(data)
scale(data, attributes)
train, test = train_test_split(data, test_size = TEST_PERCENT)


tree = tree.DecisionTreeClassifier()
tree = tree.fit(train[attributes], train[LABEL])
predict_vals = tree.predict(test[attributes])
print("Accuracy:",metrics.accuracy_score(test[LABEL], predict_vals))
