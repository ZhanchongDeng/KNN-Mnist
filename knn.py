'''
This file contains all the methods used for a KNN on the MNIST dataset.
Author: Zhanchong Deng
Date:  1/27/2020
'''
import numpy as np
import pandas as pd
import random

'''
Visualize a given entry.
'''
def toString(pic_arr):
    pic_str = ""
    # visualize one single entry
    for i in pic_arr.reshape(28,28):
        for j in i:
            if (j == 0):
                pic_str += "1"
            else:
                pic_str += "0"
        pic_str += "\n"

    return pic_str
    
'''
KNN model. Takes in file path and build a model.
'''
class KNN:
    '''
    This method reads in raw data in txt file and parse them into vectors.
    Returns a tuple of training data in numpy arrays and labels
    '''
    def loadData(self, fp):
        train_file = open(fp, 'r')
        train_file.seek(0)
        raw_strings = train_file.read().split("\n")[:-1]
        labels = [int(entry[-1:]) for entry in raw_strings]
        train_raw = [np.array(entry[0:-1].split(" ")[:-1], dtype="int") for entry in raw_strings]
        self.model = np.array(train_raw)
        self.labels = np.array(labels)

    '''
    This method uses KNN model to predict an incoming image's label by comparing to kth nearest neighbor and return the majority of the labels.
    Returns a label from 0-9.
    '''
    def predict(self, new_data, k):
        # Calculate distance from each entry of the model
        distance = np.apply_along_axis(sqr_distance, 1, self.model, new_data=new_data)
        kth_nearest = pd.DataFrame({"distance":distance, "labels":self.labels}).sort_values("distance")[:k]
        return find_mode(kth_nearest['labels'])
    
'''
Calculate square distance from given data to original data.
Returns an integer representing the distance away from the given data.
'''
def sqr_distance(cur_data, new_data):
    return np.sum((new_data - cur_data)**2)

'''
This method finds the mode of the given array. Mainly used for finding the outputting label.
Return the mode of the given data.
'''
def find_mode(labels):
    # Collect unique values
    uniques = np.unique(labels)
    # Collects the counts of each unique
    counts = [np.sum(labels == unique) for unique in uniques]
    # Find the labels with the max counts
    maxes = uniques[counts == np.amax(counts)]
    # Apply randomness if there are more than one
    return maxes[random.randint(0, len(maxes)-1)]