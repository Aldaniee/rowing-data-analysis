# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:45:02 2020

@author: Aidan Lee
@author: Tyler Brennan

"""

import csv
import copy
from sklearn.cluster import KMeans
from sklearn import svm
import random
import numpy as np
import matplotlib.pyplot as plt

DATA_NAME = "rowingdata.csv"
LABEL = "BoatPlacementImprovement"
TEST_PERCENT = 0.10
MAX_LABEL = 0             # Determined from code
GROUP_CAP = 5           # Cap for the number of groups used to determine information gain
COLORS = 'bgrcmy' # All colors for 2D plots
# In: name: file name
# Out: data: dictionary that contains all the data
# Out: data_label: array of lables or attributes for data
def get_data(name):
    global DIM
    global K
    data = {}
    attributes = []
    with open(name, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headings = next(reader)
        for i in range(len(headings)):
            data[headings[i]] = []
            attributes.append(headings[i])
        for entry in reader:
            DIM = len(entry)
            for i in range(len(entry)):
                if(entry[i] == '?'):
                    data[attributes[i]].append(entry[i])
                else:
                    data[attributes[i]].append(float(entry[i]))
    replace_unknown(data, attributes)
    return data, attributes

def data_split(input_data, test_percent):
    training_data = copy.copy(input_data)
    test_data = {}
    for key in input_data:
        test_data[key] = []
    for i in range(int(len(input_data[LABEL]) * test_percent)):
        index = random.randrange(len(training_data[LABEL]))
        for key in input_data:
            test_data[key].append(training_data[key].pop(index)) 
    return training_data, test_data

# Repalce all '?' with some value
def replace_unknown(data, attributes):
    for title in attributes:
        for i in range(len(data[title])):
            if data[title][i] == '?':
                data[title][i] = 0

# Scales all data to [0,1]
def standardize_data(input_data):
    normal_data = copy.copy(input_data)
    for key in normal_data:
        if(key != LABEL):
            min_data = float(min(normal_data[key]))
            max_data = float(max(normal_data[key])) - min_data
            for i in range(len(normal_data[key])):
                normal_data[key][i] = (float(normal_data[key][i]) - min_data) / max_data
    return normal_data

def convert_array(in_dictionary):
    dictionary = copy.copy(in_dictionary)
    dictionary.pop(LABEL)
    arr = []
    i = 0
    for key in dictionary.keys():
        arr.append([])
        for data in dictionary[key]:
            arr[i].append(data)
        i += 1
    print(arr)
    return arr

# Divides result into categories with the determining attributes (may want to change name later)
def entropy_groups(attribute, data):
    tree = []
    group = []
    over_cap = False
    for i in range(len(data[attribute])):
        val = data[attribute][i]
        if len(group) > GROUP_CAP:
            over_cap = True
            break
        elif val not in group:
            group.append(val)                           # Holds the number of features to check if over cap
            tree.append([])
            tree[group.index(val)].append(val)          # Appends the feature
        tree[group.index(val)].append(data[LABEL][i])   # Appends label associated to feature
    if over_cap:
        tree = []
        group_count = 1 / GROUP_CAP
        for i in range(GROUP_CAP):
            tree.append([])
            tree[i].append(group_count * (i + 1))       # Creates segmented dividers
        for i in range(len(data[attribute])):
            for j in tree:
                if data[attribute][i] <= j[0]:           # Determines where label belongs in dividers based on actual feature
                    j.append(data[LABEL][i])
                    break  
    return tree

# Calculates the fraction of each label present in each attribute
# This was done as binary values, so needs to be repeated with each label and must be weighted
def attribute_split(group):
    t_group = copy.deepcopy(group)
    label = []
    total_labels = [0] * (MAX_LABEL + 1)
    loop = 0
    total = 0
    for feature in t_group:
        all_attr = 0
        label.append([0] * (MAX_LABEL + 2))  # [attribute name, label fraction, non-label fraction, weight of attribute]
        label[loop][0] = feature.pop(0)
        for i in feature:
            all_attr += 1
            label[loop][int(i + 1)] += 1
            total_labels[int(i)] += 1
        label[loop].append(all_attr)
        if all_attr:
            for i in range(MAX_LABEL + 1):
                label[loop][int(i + 1)] /= all_attr
        total += all_attr
        loop += 1
    for i in range(len(total_labels)):
        total_labels[i] /= total
    for i in label:
        i[int(MAX_LABEL + 2)] /= total
    return label, total_labels

# Calculates the entropy values for each attribute
def info_gain(attribute, data):
    information_gain = 0
    system_entropy = 0
    groups = entropy_groups(attribute, data)
    final = []
    dec_groups, total_labels = attribute_split(groups)
    for i in total_labels:
        if i != 0:
            system_entropy -= (i * np.log2(i))
    for decimal in dec_groups:
        val = 0
        decimal.pop(0)
        weight = decimal.pop(len(decimal) - 1)
        for j in decimal:   # Calculates entropy for each label value for every attribute
            if j != 0:
                val -= (j * np.log2(j))
        val *= weight       # Applies the weight to the entropy to find the information gain
        final.append(val)       # final : [find label values][find feature values][0 = actual feature value, 1 = entropy value]
    information_gain = (system_entropy - sum(final))
    return information_gain

input_data, attributes = get_data(DATA_NAME)

normal_data = standardize_data(input_data)

# Split Data
training_data, test_data = data_split(normal_data, TEST_PERCENT)
attributes_without_label = copy.copy(attributes)
attributes_without_label.pop(attributes_without_label.index(LABEL))

MAX_LABEL = int(max(training_data[LABEL]))

all_info_gain = []
for attribute in attributes_without_label:
    all_info_gain.append(info_gain(attribute, training_data))


print(all_info_gain)
"""
for i in range(len(attributes_without_label)):
    print('Information gain of', attributes_without_label[i], 'is', all_info_gain[i], '\n')
"""

