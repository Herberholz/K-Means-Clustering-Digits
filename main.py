import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import copy



class Cluster():
    def __init__(self):
        self.centroid = np.random.randint(17,size=64) #current centroid
        self.preCentroid = np.zeros(64) #previous centroid
        self.label = -1

    def changeDataIndex(self, points, centroid, class_): #add new point indexs in data and the data
        self.preCentroid = copy.deepcopy(self.centroid) #save previous centroids
        self.points = points #get all points in the cluster
        self.class_ = class_ #get all classes in the cluster
        if len(points) != 0:
            self.centroid = centroid

    def mse(self): #calculate mean square error in the cluster
        l = len(self.points) #get the number of points
        if l:
            return 1, np.sum((self.points-self.centroid)**2)/l # (counting this cluster 1, mse)
        else:
            return 0,0 #if there's no point in this cluster, return 0 for not counting this cluster, and 0 for the mean square error

    def label_(self):
        if len(self.class_) !=0:
            self.label = np.bincount(self.class_).argmax() #get label for this cluster


class ClusterSet():
    def __init__(self, k, data, label):
        self.k = k # k clusters
        self.data = data #all points
        self.data_length = len(self.data)
        self.clusterSet = np.zeros(k, dtype=Cluster) #an array stores clusters
        self.label = label # array of labels
        for i in range(self.k): #initializing k clusters
            cl = Cluster()
            self.clusterSet[i] = cl


    def EuclideanDistance(self, x, y): #return an array of distances between points and a centroid
        return np.sum(np.square(x-y),axis=1)

    def run(self):

        indexCluster = np.zeros(self.k, dtype=object) #an array contains the point indexs of k clusters (size = k)

        distance = np.zeros((self.k,self.data_length)) #initializing the distance array (kxl), an array contains the distance of each point and centroid (k rows = k clusters, l columns = l points)
        #print(self.data_length)
        for i in range(self.k):
            centr = self.clusterSet[i].centroid
            distance[i] = self.EuclideanDistance(centr,self.data) #calculate Euclidean Distance between each point and seed, then store them in a 2d array (kxl)
            print(distance[i])
        closestPoints = np.argmin(distance,axis=0) #determine which cluster a point is closest to. The index and value of each element are the index of point and the index of seed. (l)

        centroids = copy.deepcopy([self.clusterSet[i].centroid for i in range(self.k)]) # initializing an array of next centroids

        for i in range(self.k):
            indexCluster[i] = np.where(closestPoints==i)[0] #get point indexs in cluster i
            if len(indexCluster[i]) != 0:
                centroids[i] = np.average(self.data[indexCluster[i]], axis=0)

        while self.isconveraged(centroids) != 1: #check if K-mean is converaged
            # print(centroids[0])
            distance = np.zeros((self.k, self.data_length))#initializing the distance array (kxl)
            for i in range(self.k):
                self.clusterSet[i].changeDataIndex(self.data[indexCluster[i]], centroids[i], self.label[indexCluster[i]])  # add point indexs to each cluster
                distance[i] = self.EuclideanDistance(self.clusterSet[i].centroid, self.data)# calculate Euclidean Distance between each point and seed, then store them in a 2d array (kxl)


            closestPoints = np.argmin(distance, axis=0)  # determine which cluster a point is closest to. The index and value of each element are the index of point and the index of seed. (l)
            for i in range(self.k):
                indexCluster[i] = np.where(closestPoints == i)[0]
                if len(indexCluster[i])!=0:
                    centroids[i] = np.average(self.data[indexCluster[i]], axis=0) #calculate new centroids for each new cluster


    def isconveraged(self, newCenters): #check if K-mean is converaged, if yes return 1; otherwise return 0
        for i in range(self.k):
            if np.array_equal(self.clusterSet[i].centroid, newCenters[i]) == 0 and np.array_equal(newCenters[i], self.clusterSet[i].preCentroid) == 0: #check if centers stop changing or if the algorithm is stuck in an oscillation.
                return 0
        return 1

    def AvgMSE(self): #calculate Average mean square Error of this cluster
        mse =0
        n = 0
        for i in range(self.k):
            n_, mse_ = self.clusterSet[i].mse()
            if n_:
                mse = mse + mse_
                n = n+n_
        if n:
            return mse/n
        else:
            return -1

    def mss(self): #calculate Mean Square Seperation
        mss_ = 0
        count = 0
        for i in range(self.k):
            # calculating the number of non-empty cluster
            n_, mse_ = self.clusterSet[i].mse()
            if n_:
                count = count + n_

            for j in range(i+1,self.k):
                if np.array_equal(self.clusterSet[i].centroid,self.clusterSet[j].centroid) == 0: #if 2 clusters are different, calculate d^2 of their centroids
                    d = np.sum(np.square(self.clusterSet[i].centroid - self.clusterSet[j].centroid))
                    mss_ = mss_+d

        return 2*mss_/(count*(count-1))



    def labelCluster(self):
        for i in range(self.k):
            self.clusterSet[i].label_()

    def predict__(self, point):
        minDis = np.inf
        n = -1
        for i in range(self.k): #calculate the smallest distance between point and centroids
            d = np.sum(np.square(self.clusterSet[i].centroid - point))
            if minDis > d:
                minDis = d
                n = i
        return self.clusterSet[n].label #return the centroid label for the smallest distance

    def predict(self, points):
        return np.array([self.predict__(point) for point in points])


#reading and split training data
data_train = pd.read_csv('optdigits.train', header=None, index_col=64)
x_train = data_train.values
y_train = data_train.index.values

#reading and split test data
data_test = pd.read_csv('optdigits.test', header=None, index_col=64)
x_test = data_test.values
y_test = data_test.index.values

# 1. You need to initialize ClusterSet (K-mean): cluster = ClusterSet(K,x_train,y_train)
# 2. You need to use 'run' function to cluster the data: cluster.run()
# 3. You need to use 'labelCluster' function to label each cluster in ClusterSet (K-mean) with y_train: cluster.labelCluster()
# 4. Use 'predict' function to predict test data. This function will return an array of labels associated with test data: cluster.predict(x_test)



## Experiment 1
mse = 9999
cluster_ = object
for i in range(5): #run 5 times to get the smallest mse
    cluster = ClusterSet(10, x_train, y_train) #initializing ClusterSet with k=10
    cluster.run() #run K-mean
    mse_ = cluster.AvgMSE()
    if mse > mse_:
        mse = mse_
        cluster_ = cluster

cluster_.labelCluster() #initializing the cluster set
y_predicted = cluster_.predict(x_test)
print("Accuracy: %s " % accuracy_score(y_test, y_predicted))
print("Average mean-square-error: %s" % cluster_.AvgMSE())
print("Mean-square-separation: %s" % cluster_.mss())
cm = confusion_matrix(y_test,y_predicted)
print(cm)
####Visualize the resulting cluster centers
for i in range(10):
    plt.figure(1)
    plt.suptitle('%d'% cluster_.clusterSet[i].label)
    img = np.reshape(cluster_.clusterSet[i].centroid,(8,8))
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    # plt.show()
    plt.axis('off')
    plt.savefig('ex1_%d_%d.jpg'% (i,cluster_.clusterSet[i].label))


#############################

# ## Experiment 2
mse = 9999
cluster_ = object
for i in range(5): #run 5 times to get the smallest mse
    cluster = ClusterSet(30, x_train, y_train)#initializing ClusterSet with k=30
    cluster.run()#run K-mean
    mse_ = cluster.AvgMSE()
    if mse > mse_:
        mse = mse_
        cluster_ = cluster

cluster_.labelCluster()#initializing the cluster set
y_predicted = cluster_.predict(x_test)
print("Accuracy: %s " % accuracy_score(y_test, y_predicted))
print("Average mean-square-error: %s" % cluster_.AvgMSE())
print("Mean-square-separation: %s" % cluster_.mss())
cm = confusion_matrix(y_test,y_predicted)
print(cm)
####Visualize the resulting cluster centers
#for i in range(30):
#    plt.figure(1)
#    plt.suptitle('%d'% cluster_.clusterSet[i].label)
#    img = np.reshape(cluster_.clusterSet[i].centroid,(8,8))
#    plt.imshow(img, cmap=plt.get_cmap('gray'))
    # plt.show()
#    plt.axis('off')
#    plt.savefig('ex2_%d_%d.jpg'% (i,cluster_.clusterSet[i].label))