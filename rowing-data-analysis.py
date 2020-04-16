# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:45:02 2020

@author: Aidan Lee
@author: Tyler Brennan

"""
import csv

DATA_NAME = "rowingdata.csv"

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
                    data[attributes[i]].append(entry[i])
    return data, attributes

data, attributes = get_data(DATA_NAME)
print(data)