# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:38:32 2019

@author: hanwe
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def load_data_set(filename):
    x = []
    y = []
    f = open(filename, 'r')
    for line in f:
        data = line.split("\t")
        x.append(data[2:-1])  # elements before the last column
        y.append(data[1])  # the last element
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=int)
    return x, y

def distance(v1,v2):
    return np.sqrt(np.sum(np.power(v1-v2,2)))

def randomCenter(x,K):
    n=np.shape(x)[1]
    center=np.mat(np.zeros([K,n]))
    for j in range(n):
        minj=np.min(x[:,j]) #min in column j
        maxj=np.max(x[:,j]) #max in column j
        rangej=float(maxj-minj) #diff between max and min
        center[:,j]=np.mat(minj+rangej*np.random.rand(K,1))   
    return center

def Kmeans(x,K):
    m=np.shape(x)[0]
    clusterAssement=np.mat(np.zeros([m,1]))
    center=randomCenter(x,K) #init center
    clusterChanged=True #whether there are elements changed in last iteration
    while clusterChanged:
        clusterChanged=False
        #for each point
        for i in range(m):
            minDist=np.inf #min distance between point and center
            minIndex=-1 #the index of nearest center
            #for each center
            for j in range(K):
                dist=distance(center[j,:], x[i,:])
                if dist<minDist:
                    minDist=dist
                    minIndex=j 
            if clusterAssement[i,0]!=minIndex:    #whether the point changes cluster
                clusterChanged=True 
            clusterAssement[i]=minIndex
        #calculate the new center
        for point in range(K):
            ptsInClust=x[np.nonzero(clusterAssement[:,0].A==point)[0]] #find the points for each cluster
            center[point,:] = np.mean(ptsInClust, axis = 0)   #calculate the new center
    return center,clusterAssement 
    
def diff(clusters,y,K):
    corresponding=np.zeros((K, K), dtype=np.int)
    m=np.shape(y)[0]
    for i in range(m):
        corresponding[int(clusters[i])][int(y[i])-1]+=1
        res=0;
    for i in range(K):
        res+=np.max(corresponding[i,:])
    accuracy=res/m
    return accuracy,corresponding

def calcRand(clusters,y):
    m=np.shape(y)[0]
    C=np.zeros((m,m), dtype=np.int)
    Y=np.zeros((m,m), dtype=np.int)
    for i in range(m):
        for j in range(m):
            if clusters[i]==clusters[j]:
                C[i][j]=1
    for i in range(m):
        for j in range(m):
            if y[i]==y[j]:
                Y[i][j]=1
    res=0
    for i in range(m):
        for j in range(m):
            if C[i][j]==Y[i][j]:
                res=res+1
    return res/(m*m)

x,y = load_data_set('cho.txt')
K=max(y) #number of clusters
center,clusters=Kmeans(x,K)
y=y.reshape(-1,1)
accuracy,corresponding=diff(clusters,y,K)
print("accuracy:",accuracy)
print(corresponding)
rand=calcRand(clusters,y)
print("rand:",rand)
