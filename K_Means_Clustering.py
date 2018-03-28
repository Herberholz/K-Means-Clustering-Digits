# Cody Herberholz
# CS445 HW5 K Means Clustering
# in order to properly run program just make sure to run program via IDE or Linux normally
# When needing to change cluster number, make sure to change the Global cluster number just below

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Global cluster number
K = 10

# Takes in x_train or x_test data, center of clusters, and distance array to be filled
# Purpose of function is to calculate euclidean distance and find out the minimum distance
#   which specifies which cluster it belongs to
def classify_points(x, centroid, distance):
    for i in range(K):
        distance[i] = ((x - centroid[i]) ** 2).sum(axis=1)

    closest = np.argmin(distance, axis=0)

    return closest

# With established cluster data entries, a new centroid is created
def establish_centroid(x_train, closest, count):
    new_centroid = np.zeros((K, 64))
    size = len(new_centroid[0])

    for i in range(len(x_train)):
        index = closest[i]
        count[index] += 1 # keeps count of how many datapoints each cluster has
        new_centroid[index] += x_train[i]

    for i in range(K):
        div = count[i]
        for j in range(size):
            if new_centroid[i][j] != 0:
                new_centroid[i][j] = new_centroid[i][j] / div
                new_centroid[i][j] = int(new_centroid[i][j])

    return new_centroid

# Calculates the average mean square error and returns it
def avg_mean_squared_error(distance, elements, x_train, new_centroid, closest):
    mse = np.zeros(K)
    empty = 0

    for i in range(K):
        for j in range(len(distance[0])):
            index = closest[j]
            mse[index] += ((x_train[j] - new_centroid[index]) ** 2).sum(axis=0)

    for i in range(K):
        if elements[i] != 0:
            mse[i] = mse[i] / elements[i]
        else:
            empty += 1
    avg = np.sum(mse) / (K - empty)

    return avg

# Calculates the mean square separation and returns it
def mean_square_separation(centroid):
    mss = 0

    for i in range(K-1):
        for j in range(i+1, K):
            equal = np.array_equal(centroid[i], centroid[j])
            if equal == False:
                mss += ((centroid[i] - centroid[j]) ** 2).sum(axis=0)

    div = K * (K-1) / 2
    return mss / div

# Creates data based off of the optdigits.train file, then randomly generates K
#   centroids that will be adjusted over time until they stop changing. After 5
#   runs of this the final centroid with the smallest mse is chosen to be returned
def train(final_mse):
    data = pd.read_csv("optdigits.train", header=None, index_col=64)
    x_train = data.values

    for n in range(5):
        centroid = np.zeros((K, 64))
        distance = np.zeros((K, len(x_train)))
        elements = np.zeros(K)
        count = 0

        for i in range(K):
            centroid[i] = np.random.randint(17, size=64)

        closest = classify_points(x_train, centroid, distance)
        new_centroid = establish_centroid(x_train, closest, elements)
        equal = np.array_equal(centroid, new_centroid)

        while equal == False and count < 300:
            elements = np.zeros(K)
            centroid = new_centroid
            closest = classify_points(x_train, centroid, distance)
            new_centroid = establish_centroid(x_train, closest, elements)
            count += 1
            equal = np.array_equal(centroid, new_centroid)

        mse = avg_mean_squared_error(distance, elements, x_train, new_centroid, closest)
        if mse < final_mse:
            final_mse = mse
            final_centroid = new_centroid


    mms = mean_square_separation(final_centroid)
    print("MMS: ", mms)
    print("MSE: ", final_mse)
    return final_centroid

# Uses the chosen centroid on the test data
# Prints out accuracy, confusion matrix, and visual data
def test(centroid):
    data= pd.read_csv("optdigits.test", header=None, index_col=64)
    x_test = data.values
    y_test = data.index.values

    size = len(x_test)
    distance = np.zeros((K, size))
    conf_matrix = np.zeros((10, K))

    closest = classify_points(x_test, centroid, distance)
    for i in range(size):
        prediction = closest[i]
        true = y_test[i]
        conf_matrix[true][prediction] += 1

    print(conf_matrix)
    print(metrics.accuracy_score(y_test, closest))

    for i in range(K):
        image = np.reshape(centroid[i], (8, 8))
        plt.imshow(image, cmap = plt.get_cmap('gray'))
        plt.savefig('EX2_Centroid_%d.jpg' % i)

    return

# Controls flow of K means program
def main():
    final_mse = 80000
    final_centroid = np.zeros((K, 64))

    final_centroid = train(final_mse)
    test(final_centroid)
    return

# Runs program
main()
