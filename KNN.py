import numpy as np
import pandas as pd
import math
import operator
import random
from sklearn.model_selection import train_test_split


# find euclidian distance
def euclideanDistance(instance1, instance2, dimension):
    distance = 0
    for x in range(dimension):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

#find the k-most similar neighbours
def findNeighbours(train_data, test_data_instance, k):
    distances = []
    length = len(test_data_instance) - 1
    for x in range(len(train_data)):
        dis = euclideanDistance(test_data_instance, train_data[x], length)
        distances.append((train_data[x],dis))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(iter(classVotes.items()), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# load data into dataframe
data = pd.read_csv("knndata.csv", delimiter=',', names=['X1', 'X2', 'X3', 'X4', 'label'])

# Split the data into training and testing
train_data, test_data = train_test_split(data, test_size=0.2)
np_train = np.array(train_data[:-2])
np_test = np.array(test_data[:-2])

# generate predictions
predictions = []
k = 3
for x in range(len(np_test)):
    neighbors = findNeighbours(np_train, np_test[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(np_test[x][-1]))
accuracy = getAccuracy(np_test, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

